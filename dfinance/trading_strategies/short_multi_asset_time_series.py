"""
TODO: Expected data format is 404 column 
correlation matrix built elsewhere. Need to
consider how to more cheaply create features
from just a list of asset prices (all in one training).
"""
from datetime import datetime
import math

import pandas as pd
import numpy as np
from tqdm import tqdm

from dfinance.utils import ledger

from scipy.stats import pearsonr
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler


def now():
    return datetime.now()


class TimeSeriesModel:
    """
    Builds time series model dataframe.
    """

    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.ml = LGBMRegressor(n_estimators=200, num_leaves=300, learning_rate=0.09)
        self.stats = {}

    def build(self, target_asset):
        # ITS ON YOU TO ENSURE YOU HAVE A HEALTHLY DF BEFORE THIS DROPNA()
        x_train = self.get_features(self.train_set, target_asset).dropna()
        y_train = self.train_set.loc[x_train.index][target_asset]
        x_test = self.get_features(self.test_set, target_asset).dropna()
        y_test = self.test_set.loc[x_test.index][target_asset]

        self.ml.fit(x_train, y_train)
        # self.ml.predict(x_train) => np.array
        # print(y_train)
        self.stats["train-in-sample-accuracy"] = self.calc_stats(
            self.ml.predict(x_train), y_train
        )
        self.stats["train-out-sample-accuracy"] = self.calc_stats(
            self.ml.predict(x_test), y_test
        )
        return self

    def online_preprocess(self, row, target_asset):
        # process row like dataframe
        r = pd.DataFrame([row.to_dict()])
        X = self.get_features(r, target_asset)
        return self.ml.predict(X)[0]

    def calc_stats(self, y_hat, actual):
        mse = np.mean(actual - y_hat) ** 2
        cv = pearsonr(y_hat, actual)[0]
        return {"mse": mse, "cv": cv}

    def get_features(self, df, target_asset: str):
        X = df[[i for i in df.columns if target_asset not in i and i != "ts"]]
        return X


def filter_on_datetime_index(df, start_time, end_time):
    # between_time seems broken AF
    tmp = df.copy()
    tmp["ts"] = tmp.index
    slice = tmp[(tmp.ts >= start_time) & (tmp.ts <= end_time)]
    return slice


class ShortOnlyMultiAssetStrategy:
    """
    A strategy that uses as dataframe format that will work with model
    to generate expected future price predictions.

    This model treats the problem as cold on each target data set
    and retrains the model from scratch every time, just on the dataset.

    Short-Only because I'm a pessimist.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        ledger: ledger.Ledger,
        trade_cost: float,
        tradable_assets: list,
        capital: int,
    ):
        # objects
        self.df = features_df
        self.prices = prices_df
        self.ledger = ledger
        self.ALLOC_PCT = 0.10

        # params
        self.trade_cost = trade_cost
        self.tradable_assets = tradable_assets

        # fund the ledger
        self.capital = capital
        self.starting_capital = capital
        self.positions = {}
        # self.ledger.add_transaction(
        #     ticker="USD", shares=capital, price=1, exec_time=min(self.df.index)
        # )

    def get_capital(self, amount, purpose):
        self.capital -= amount
        # TODO: store purpose and get capital
        return amount

    def give_back_capital(self, amount, source):
        self.capital += amount
        return amount

    def test(self, start_date, end_date):
        # TODO, make target_Asset = multiple
        test_range = filter_on_datetime_index(self.df, start_date, end_date)
        train_range = self.df.loc[~self.df.index.isin(test_range.index)]
        # split data into train and test
        train_model = TimeSeriesModel(train_range, test_range).build(
            # TODO: make this all assets
            target_asset=self.tradable_assets[0]
        )
        # for each day in our time series (assumes it is sorted)
        for ts, row in tqdm(test_range.iterrows()):
            for target_asset in self.tradable_assets:
                pred = train_model.online_preprocess(row, target_asset)
                # first sell the position if need be to free up capital
                if pred > 0.01 and self.positions.get(target_asset) != None:
                    shares = self.positions.get(target_asset)
                    cur_price = self.prices.loc[ts, target_asset]
                    freed_up_liquidity = shares * cur_price
                    self.give_back_capital(
                        freed_up_liquidity, source=f"exit {target_asset}"
                    )
                    self.positions[target_asset] = None
                    self.ledger.add_transaction(
                        ticker=target_asset,
                        shares=-shares,
                        price=cur_price,
                        exec_time=ts,
                        details={"type": "buy"},
                        # broadcast=True,
                    )

                # decide its worth entering a position
                if pred < -0.01:

                    price_per_share = self.prices.loc[ts, target_asset]
                    alloc = min(
                        [self.ALLOC_PCT * self.starting_capital, self.capital]
                    )  # get allocation percent or min capital available
                    shares_to_short = math.floor(alloc / price_per_share)
                    position_capital_required = shares_to_short * price_per_share

                    _ = self.get_capital(
                        position_capital_required, purpose=f"short {target_asset}"
                    )
                    if shares_to_short == 0:
                        continue
                    else:
                        self.ledger.add_transaction(
                            **{
                                "ticker": target_asset,
                                "shares": -shares_to_short,
                                "price": price_per_share,
                                "exec_time": ts,
                                "details": {"type": "sell"},
                            },
                            # broadcast=True
                        )
                        self.positions[target_asset] = -shares_to_short
