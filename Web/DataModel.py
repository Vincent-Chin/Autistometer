"""

ToDo:
    * cumulative PL (daily, weekly, monthly) | equal-weighted / vote-share-weighted | SPY-buy-hold
    * Line chart, scalable with bokeh
    * entries list (by date) with (vote share, polarity, direction, pls)

"""

from Engine import sqlite_db
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

data = None     # This houses our data


class DataModel:

    def __init__(self, db_path='../autistometer.db'):
        self.sentiments, self.pnls, self.stock_perfs = self.process_sentiments(sqlite_db.fetch_all_sentiments(db_path))
        return

    def process_sentiments(self, sentiments):
        """
        This function calculates and caches PLs from the sentiments object for quick delivery to web requests.
        :return: pd.DataFrame
        """

        df = pd.DataFrame()
        pnls = {}
        stock_perfs = pd.DataFrame()
        if sentiments is None or len(sentiments) == 0:
            pass
        else:
            for calendar_date in sentiments:
                signals = pd.DataFrame(sentiments[calendar_date]).transpose().reset_index()
                signals = signals.rename(columns={'index': 'ticker'})
                signals['calendar_date'] = calendar_date
                df = df.append(signals)
            df = df.set_index(['calendar_date', 'ticker'])
            df['portfolio_weight'] = df['vote_share'] / df.groupby('calendar_date').sum()['vote_share']

            excluded_columns = ['vote_share', 'spy_correlation', 'avg_volume', 'return_1d', 'return_5d', 'return_20d',
                                'portfolio_weight']
            wm = lambda x: np.average(x, weights=df.loc[x.index, "portfolio_weight"])
            pnls['EQUAL'] = df.groupby(['calendar_date']).mean().drop(excluded_columns, axis=1)
            pnls['WEIGHTED'] = df.groupby(['calendar_date']).agg(wm).drop(excluded_columns, axis=1)
            stock_perfs = df.groupby(['ticker']).mean().drop(excluded_columns, axis=1)

            for col in df.columns:
                if col.startswith('profit'):
                    pnls['EQUAL']['cum' + col] = np.cumsum(pnls['EQUAL'][col])
                    pnls['WEIGHTED']['cum' + col] = np.cumsum(pnls['WEIGHTED'][col])

        return df, pnls, stock_perfs

    # TODO: buzz-adjusted dates (scale PL by total votes on the day vs avg)
    # TODO: descriptive statistics (win rate, win ratio, sharpe, etc)
    # TODO: market benchmark, vix benchmark


def debug():
    db_path = 'autistometer.db'
    global data
    data = DataModel(db_path)

    # plot examples
    data.pnls['EQUAL'][['cumprofit_1d', 'cumprofit_5d', 'cumprofit_20d']].plot(
        title='EQUAL-WEIGHT').axhline(y=0, color='r')
    data.pnls['WEIGHTED'][['cumprofit_1d', 'cumprofit_5d', 'cumprofit_20d']].plot(
        title='VOTESHARE-WEIGHT').axhline(y=0, color='r')
    return