"""
Heavily sourced from: https://www.sqlitetutorial.net/sqlite-python/create-tables/
"""

import sqlite3
from sqlite3 import Error
import datetime as dt
import dateutil.parser
import pandas as pd


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
    return


def initialize_db(db_path):

    print(dt.datetime.now(), "Initializing database at", db_path)

    database = db_path

    sql_create_sentiments_table = """ CREATE TABLE IF NOT EXISTS sentiments (
                                        ticker_id int,
                                        calendar_date string,
                                        positive real NOT NULL,
                                        negative real NOT NULL,
                                        polarity real NOT NULL,
                                        vote_share real NOT NULL,
                                        
                                        vix_correlation real,
                                        spy_correlation real,
                                        avg_volume real,
                                        direction int,
                                        return_1d real,
                                        return_5d real,
                                        return_20d real,
                                        profit_1d real,
                                        profit_5d real,
                                        profit_20d real,
                                        
                                        FOREIGN KEY (ticker_id) REFERENCES tickers (id),
                                        PRIMARY KEY (ticker_id, calendar_date)
                                        
                                    ); """

    sql_create_tickers_table = """CREATE TABLE IF NOT EXISTS tickers (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL
                                );"""

    sql_create_benchmarks_table = """CREATE TABLE IF NOT EXISTS benchmarks (
                                        ticker_id int,
                                        calendar_date string,
                                        open real,
                                        close real,
                                        volume real,
                                        
                                        FOREIGN KEY (ticker_id) REFERENCES tickers (id),
                                        PRIMARY KEY (ticker_id, calendar_date)                                        
                                    );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_sentiments_table)

        # create tasks table
        create_table(conn, sql_create_tickers_table)

        # create benchmarks table
        create_table(conn, sql_create_benchmarks_table)

    else:
        print("Error! cannot create the database connection.")
    return


def store_sentiments(sentiments, calendar_date, db_path):
    """

    :param sentiments: dictionary with dictionary of values; eg.
        {'SPY': {'positive': 15793.0, 'negative': 31478.0, 'vote_share': 0.4162975226990515,
                'polarity': -0.33181020075733536, 'spy_correlation': 0.9999999999999999,
                'avg_volume': 75085130.78968254, 'direction': -1.0},
         'CCL': {'positive': 1563.0, 'negative': 5308.0, 'vote_share': 0.060510255303784205,
                'polarity': -0.5450443894629603, 'spy_correlation': -0.26996905002944926,
                 'avg_volume': 5524607.464285715, 'direction': -1.0}
        }
    :return:
    """
    target_date = str(dateutil.parser.parse(calendar_date).date())

    with create_connection(db_path) as conn:
        for ticker in sentiments:
            # identify ticker.  if ticker doesn't exist, insert
            cursor = conn.execute("SELECT id FROM tickers where name='" + ticker + "' LIMIT 1")
            rows = cursor.fetchall()
            if len(rows) < 1:
                conn.execute("INSERT INTO tickers (`name`) VALUES ('" + ticker + "')")
                cursor = conn.execute("SELECT id FROM tickers where name='" + ticker + "' LIMIT 1")
                rows = cursor.fetchall()
                ticker_id = rows[0][0]
            else:
                ticker_id = rows[0][0]

            columns = ', '.join("`" + str(x).replace('/', '_') + "`" for x in sentiments[ticker].keys())
            values = ', '.join("'" + str(x).replace('/', '_') + "'" for x in sentiments[ticker].values())
            query = "REPLACE INTO %s (`ticker_id`, `calendar_date`, %s ) VALUES ( '%s', '%s', %s );" % (
                'sentiments', columns, ticker_id, target_date, values)
            query = query.replace("'None'", ":null")
            conn.execute(query, {'null': None})
    return


def fetch_sentiments(calendar_date, db_path):
    target_date = str(dateutil.parser.parse(calendar_date).date())
    with create_connection(db_path) as conn:
        query = ''.join([
            "SELECT * FROM tickers, sentiments where calendar_date =='",
            target_date, "' AND tickers.id == sentiments.ticker_id"])
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        results = {}
        columns = [column[0] for column in cursor.description]
        for row in rows:
            this_row = dict(zip(columns, row))
            stock_ticker = this_row['name']
            del this_row['id']
            del this_row['ticker_id']
            del this_row['name']
            del this_row['calendar_date']
            results[stock_ticker] = this_row

        return results


def fetch_all_sentiments(db_path, from_date=None, no_perf_only=False):
    with create_connection(db_path) as conn:
        added = ""
        if from_date:
            added += " and calendar_date >= %s" % str(from_date)
        if no_perf_only:
            added += " and sentiments.return_20d isnull"
        query = ''.join([
            "SELECT * FROM tickers, sentiments where tickers.id == sentiments.ticker_id", added])
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        results = {}
        columns = [column[0] for column in cursor.description]
        for row in rows:
            this_row = dict(zip(columns, row))
            stock_ticker = this_row['name']
            calendar_date = this_row['calendar_date']

            del this_row['id']
            del this_row['ticker_id']
            del this_row['name']
            del this_row['calendar_date']

            if calendar_date not in results:
                results[calendar_date] = {}
            results[calendar_date][stock_ticker] = this_row

        for calendar_date in list(results.keys()):
            if len(results[calendar_date]) == 0:
                results.pop(calendar_date)

        return results


def fetch_benchmark(ticker, start_date, end_date, db_path):
    start_date = str(dateutil.parser.parse(str(start_date)).date())
    end_date = str(dateutil.parser.parse(str(end_date)).date())

    result = pd.DataFrame()
    with create_connection(db_path) as conn:
        # first get the ticker id
        query = "SELECT id FROM tickers WHERE name = '%s' LIMIT 1" % ticker
        rows = conn.execute(query).fetchall()
        if len(rows) < 1:
            pass
        else:
            ticker_id = rows[0][0]
            query = (
                "SELECT * FROM benchmarks where ticker_id = '%s' AND calendar_date >='%s' AND calendar_date <= '%s'"
                % (ticker_id, start_date, end_date))
            result = pd.read_sql(query, conn, index_col='calendar_date')
            result.drop('ticker_id', axis=1, inplace=True)
    return result


def store_benchmark(db_path, ticker, history):
    with create_connection(db_path) as conn:
        # first get the ticker id
        query = "SELECT id FROM tickers WHERE name = '%s' LIMIT 1" % ticker
        rows = conn.execute(query).fetchall()
        if len(rows) < 1:
            conn.execute("INSERT INTO tickers (`name`) VALUES ('" + ticker + "')")
            cursor = conn.execute("SELECT id FROM tickers where name='" + ticker + "' LIMIT 1")
            rows = cursor.fetchall()
            ticker_id = rows[0][0]
        else:
            ticker_id = rows[0][0]

        # then format the history into our benchmark format
        history = history[['open', 'close', 'volume']].copy()
        history['ticker_id'] = ticker_id
        history.index = history.index.date
        history.index.name = 'calendar_date'

        # delete any overlap
        dquery = "DELETE FROM benchmarks WHERE ticker_id = '%s' AND calendar_date >= '%s' AND calendar_date <= '%s'" % (
            ticker_id, history.index[0], history.index[-1])
        conn.execute(dquery)

        # finally, store the info
        history.to_sql('benchmarks', conn, if_exists='append')
    return
