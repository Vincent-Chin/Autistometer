"""
 Connecting to Reddit's API: https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
 Praw API: https://praw.readthedocs.io/en/latest/getting_started/authentication.html

 1. Connect to reddit's API
 2. Request threads (best/top/controversial/gilded)
 3. Get Author, Title, Likes, Comments
 4. Potentially rank threads by author's autism score (upvotes within wallstreetbets only)
 5. Parse text and comments (recursively), and process using NLP to identify tickers or companies
 6. Simultaneously attempt to assign to bull or bear

"""

import praw
from psaw import PushshiftAPI
import wget
import os
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from urllib.parse import unquote
from nltk import FreqDist
from nltk.corpus import words
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re
import datetime as dt
import dateutil.parser
import pytz
from Engine import sqlite_db as sdb
import pandas_datareader as pdr
import multiprocessing as mp

from Engine.credentials import *
from Engine.NestedPool import NestedPool

pd.set_option('display.max_columns', 50)

# globals
db_path = "autistometer.db"


def do_nltk_installations():
    """
    This is a run-once.  Sets up the operating system to support the nltk toolkit.
    :return:
    """
    nltk.download()
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    return


def get_ticker_set():
    """
    This downloads the currently known stock ticker list from Quandl.
    :return:
    """
    url = 'https://s3.amazonaws.com/quandl-production-static/end_of_day_us_stocks/ticker_list.csv'
    if not os.path.exists('Engine/Inputs/ticker_list.csv'):
        filename = wget.download(url, out='Engine/Inputs/ticker_list.csv')
    else:
        filename = 'Engine/Inputs/ticker_list.csv'
    tickers = pd.read_csv(filename)
    ticker_set = set(tickers.Ticker.str.upper())
    return ticker_set


def get_ticker_excluded_words(n_most_common=1000):
    """
    This generates a frequency list of the most well-known English words and adds them to an exclusion-set,
    which we can use to ignore false ticker hits.

    :param n_most_common:
    :return:
    """
    # exclude wsb keywords and common abbreviations
    wsb_keywords = {'shitpost', 'discussion', 'fundamentals', 'technicals', 'stocks', 'options',
                    'futures', 'gain', 'loss', 'storytime', 'satire', 'earnings', 'thread', 'daily', 'discussion',
                    'mods',
                    
                    # commonly mistakable as tickers
                    'YOLO', 'DD', 'USD', 'USA', 'GDP', 'CEO', 'PM', 'AI', 'CASH', 'EDIT', 'F', 'CDC', 'RH', 'TD',
                    'IRS', 'IPO', 'CPA', 'IRL'
                    
                    # eliminate all one-letter tickers
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z'
                    }

    # exclude the top 2000 words of the english language from consideration
    frequency_list = FreqDist(i for i in words.words())
    stop_set = {i[0] for i in frequency_list.most_common(n_most_common)}

    # return a unified set
    exclude_set = wsb_keywords.union(stop_set)
    return exclude_set


def get_sentiment_dictionary():
    # url = "https://drive.google.com/file/d/12ECPJMxV2wSalXG8ykMmkpa1fq_ur0Rf/view?usp=sharing"
    # if os.path.exists('loughran_mcdonald.csv'):
    #    os.remove('loughran_mcdonald.csv')
    # filename = wget.download(url, out='loughran_mcdonald.csv')
    filename = 'Engine/Inputs/loughran_mcdonald.csv'
    lmframe = pd.read_csv(filename)

    sentiment_dict = {
        'Positive': lmframe[lmframe.Positive > 0].Word.str.lower().to_list(),
        'Negative': lmframe[lmframe.Negative > 0].Word.str.lower().to_list(),
        'Negation': [
            "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "ain't", "aren't",
            "can't", "couldn't", "daren't", "didn't", "doesn't", "dont", "hadnt", "hasnt", "havent", "isnt",
            "mightnt", "mustnt", "neither", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
            "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere", "oughtnt",
            "shant", "shouldnt", "wasnt", "werent", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without",
            "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite", "no", "nobody"]
    }

    # add special words
    sentiment_dict['Positive'] += ['call', 'calls', 'CALL', 'CALLS']
    sentiment_dict['Negative'] += ['put', 'puts', 'PUT', 'PUTS']

    return sentiment_dict


def tokenize(text: str):
    """
    This converts a text into individual tokens, by first applying several transformations that strip away
    hyperlinks and html.  It also detects all-caps sentences and converts them to lower-case, to avoid
    false ticker triggerings.  Finally, it lemmatizes the sentences to pair words with their parts-of-speech
    for better recognition.

    :param text:
    :return:
    """
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)  # remove URLs
    soup = BeautifulSoup(text, features="html.parser")
    stripped_text = unquote(soup.get_text())    # remove html tags, url-encoding; preserve case

    # we're going to tokenize into sentences, which we'll sub-parse into phrases and correct the casing for.
    # then, with each sentence we'll tag according to the part of speech
    sentences = nltk.sent_tokenize(stripped_text)
    lemmatized_tokens = []
    lemmatizer = WordNetLemmatizer()
    for s in sentences:
        # split into phrases and convert casing to lower if all caps
        phrases = re.split(r'(.,)', s)
        this_sentence = ""
        for p in phrases:
            if p.isupper():
                this_sentence += p.lower()
            else:
                this_sentence += p

        unlemmatized_tokens = pos_tag(nltk.word_tokenize(this_sentence))
        for word, tag in unlemmatized_tokens:
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_word = lemmatizer.lemmatize(word, pos)
            if lemmatized_word.isalnum():
                lemmatized_tokens.append(lemmatized_word)

    return lemmatized_tokens


def identify_relevant_tickers(found_tickers: dict, lemmatized_tokens: list, ticker_set: set, exclude_set: set):
    # hunt through tokens for matching tickers
    n_found = 0
    for token in lemmatized_tokens:
        if token in exclude_set:
            continue
        elif token in ticker_set:
            n_found += 1
            if token in found_tickers:
                found_tickers[token] += 1
            else:
                found_tickers[token] = 1
    return found_tickers, n_found


def identify_sentiment(lemmatized_tokens: list, sentiment_dict: dict):
    pos_count = 0
    neg_count = 0
    word_count = len(lemmatized_tokens)

    for i in range(0, word_count):
        # assess negative sentiment from the token
        if lemmatized_tokens[i].lower() in sentiment_dict['Negative']:
            neg_count += 1
        # assess positive sentiment from the token, but add negativity when we find negation words
        if lemmatized_tokens[i].lower() in sentiment_dict['Positive']:
            neg_found = 0
            for n in range(0, min(3, i)):
                if lemmatized_tokens[n].lower() in sentiment_dict['Negation']:
                    neg_found += 1
            if neg_found > 0:
                neg_count += 1
            else:
                pos_count += 1
    return [word_count, pos_count, neg_count]


def process_post(post, ticker_set, exclude_set, sentiment_dict):
    pstart = dt.datetime.now()
    # print("   ", pstart, "Processing post:", post.name)
    # do a time check to keep only the posts since the last run / update existing posts
    # post.created_utc
    # print(post.subreddit_id, post.id, post.name, post.title)
    # print(post.author, post.author_flair_text, post.author_fullname, post.author_premium)
    # print(post.score, post.num_comments)

    # ignore posts without at least one up-vote
    sentiments = {}
    if post.score > 1:
        found_tickers = {}
        found_sentiment = {'p': 0, 'n': 0}  # total word count, pos word count, neg word count

        # go through the title and text to identify related tickers
        post_tokens = tokenize(post.title + ";;" + post.selftext)
        found_tickers, n_found = identify_relevant_tickers(found_tickers, post_tokens, ticker_set, exclude_set)
        post_sentiment = identify_sentiment(post_tokens, sentiment_dict)

        # with the post sentiment, modify the found sentiment by the raw numbers multiplied by upvotes
        pn = post_sentiment[1] + post_sentiment[2]
        if pn > 0:
            found_sentiment['p'] += post_sentiment[1] / pn * post.score
            found_sentiment['n'] += post_sentiment[2] / pn * post.score

        # identify all comments - leave unprocessed MoreComments if we get an error
        try:
            post.comments.replace_more(limit=None)
        except Exception as e:
            print(post, "Too many additional comments.  Skipping:", e)
        all_comments = [x for x in post.comments.list() if not isinstance(x, praw.models.reddit.more.MoreComments)]

        # limit to top 95% of posts
        all_comments.sort(key=lambda t: t.score, reverse=True)

        # find total score
        tscore = 0
        for x in range(0, len(all_comments)):
            tscore += all_comments[x].score

        # identify post coverage -> we only keep the top 95% of posts
        if tscore > 0:
            cscore = 0
            x = 0
            for x in range(0, len(all_comments)):
                cscore += all_comments[x].score
                if cscore / tscore >= 0.95:
                    break
            comments = all_comments[0:x]

            for comment in comments:
                # ignore comments without at least one up-vote
                if comment.score > 1:
                    comment_tokens = tokenize(comment.body)
                    found_tickers, n_found = identify_relevant_tickers(found_tickers, comment_tokens,
                                                                       ticker_set, exclude_set)
                    comment_sentiment = identify_sentiment(comment_tokens, sentiment_dict)
                    pn = comment_sentiment[1] + comment_sentiment[2]
                    if pn > 0:
                        found_sentiment['p'] += comment_sentiment[1] / pn * comment.score
                        found_sentiment['n'] += comment_sentiment[2] / pn * comment.score

            # now that the post has been combed, assign bull/bear votes to each symbol detected in ratio of
            # mentions in the post.
            tick_count = 0
            for ticker in found_tickers:
                tick_count += found_tickers[ticker]

            if tick_count > 0:
                for ticker in found_tickers:
                    if ticker not in sentiments:
                        sentiments[ticker] = (found_tickers[ticker] / tick_count * found_sentiment['p'],
                                              found_tickers[ticker] / tick_count * found_sentiment['n'])
                    else:
                        sentiments[ticker] = (
                            sentiments[ticker][0] + found_tickers[ticker] / tick_count * found_sentiment['p'],
                            sentiments[ticker][1] + found_tickers[ticker] / tick_count * found_sentiment['n'])
        pstop = dt.datetime.now()
        # print("   ", pstop, "Finished post:", post.name, "Total Time:", pstop - pstart)
    return sentiments


def get_history(symbol, start_date, end_date):
    start_date = dateutil.parser.parse(str(start_date)).date()
    end_date = dateutil.parser.parse(str(end_date)).date()

    # first we'll check the database benchmarks table, to see if our data is within
    history = sdb.fetch_benchmark(symbol, start_date, end_date, db_path)
    if len(history) > 0:
        hstart = dateutil.parser.parse(str(history.index[0])).date()
        hend = dateutil.parser.parse(str(history.index[-1])).date()
    else:
        hstart = end_date
        hend = dt.datetime(1970, 1, 1).date()

    if hstart <= start_date and hend >= end_date:
        pass
    else:
        # if it's not, request the data from tiingo, then store it and retrieve it again for conformant formatting
        gap_start = start_date if start_date < hstart else hend
        try:
            history = pdr.get_data_tiingo(symbol, start=gap_start, end=end_date, api_key=tiingo_api_key).loc[symbol]
            sdb.store_benchmark(db_path, symbol, history)
            history = sdb.fetch_benchmark(symbol, start_date, end_date, db_path)
        except Exception as e:
            print(e)

    return history


def populate_performance(calendar_date, exclude_weekends, polarity, this_hist, spy_hist, vix_hist):

    ret_dict = {
        'return_1d': None,
        'profit_1d': None,
        'return_5d': None,
        'profit_5d': None,
        'return_20d': None,
        'profit_20d': None,
        'spy_correlation': None,
        'direction': None,
        'vix_correlation': None,
        'avg_volume': None,
    }

    if not(len(this_hist) > 0 and len(spy_hist) > 0):
        return ret_dict
    else:
        tdate = dateutil.parser.parse(str(calendar_date)).date()
        if not exclude_weekends:
            fdate1 = tdate + dt.timedelta(days=0)
            fdate5 = tdate + dt.timedelta(days=4)
            fdate20 = tdate + dt.timedelta(days=19)
        else:
            fdate1 = np.busday_offset(tdate, 0, roll='forward')
            fdate5 = np.busday_offset(tdate, 4, roll='forward')
            fdate20 = np.busday_offset(tdate, 19, roll='forward')

        spy_correlation = spy_hist.close.corr(this_hist.close)
        vix_correlation = vix_hist.close.corr(this_hist.close)
        avg_volume = this_hist.volume.rolling(60).mean()[-1]    # this is calculated twice, but oh well.

        direction = np.sign(polarity) * np.sign(spy_correlation) if np.abs(spy_correlation) > 0.3 else np.sign(polarity)
        ret_dict.update(dict(spy_correlation=spy_correlation, direction=direction, vix_correlation=vix_correlation,
                             avg_volume=avg_volume))

        # because our posts are computed from 5AM NY time (prior trading day) until 5AM NY time (current trading day),
        # today's close is actually supposed to be yesterday's close [idx - 2].
        # forward date is already adjusted above to include today in the adjustment.
        today_close = this_hist[this_hist.index <= str(tdate)].iloc[-2].close
        next_data = this_hist[this_hist.index > str(tdate)]
        if len(next_data) > 0:
            next_open = next_data.iloc[0].open
            basis_price = next_open
            if len(this_hist[this_hist.index >= str(fdate1)]) > 0:
                forward_close = this_hist[this_hist.index >= str(fdate1)].iloc[0].close
                ret_dict['return_1d'] = (forward_close - basis_price) / basis_price
                ret_dict['profit_1d'] = ret_dict['return_1d'] * ret_dict['direction']
            if len(this_hist[this_hist.index >= str(fdate5)]) > 0:
                forward_close = this_hist[this_hist.index >= str(fdate5)].iloc[0].close
                ret_dict['return_5d'] = (forward_close - basis_price) / basis_price
                ret_dict['profit_5d'] = ret_dict['return_5d'] * ret_dict['direction']
            if len(this_hist[this_hist.index >= str(fdate20)]) > 0:
                forward_close = this_hist[this_hist.index >= str(fdate20)].iloc[0].close
                ret_dict['return_20d'] = (forward_close - basis_price) / basis_price
                ret_dict['profit_20d'] = ret_dict['return_20d'] * ret_dict['direction']
        return ret_dict


def process_calendar_date(calendar_date, exclude_weekends=True, min_vote_share=.02,
                          min_avg_volume=1e6, num_reserved_cores=0):

    # if date is processed, skip.
    pstart = dt.datetime.now()
    if not os.path.exists(db_path):
        sdb.initialize_db(db_path)
    else:
        existing_sentiments = sdb.fetch_sentiments(calendar_date, db_path)
        if len(existing_sentiments) > 0:
            print(pstart, "Skipping", calendar_date, ".  Sentiment already exists.")
            return

    print(pstart, "Processing", calendar_date)

    # 0. get tickers and excluded keywords
    ticker_set = get_ticker_set()
    exclude_set = get_ticker_excluded_words()
    sentiment_dict = get_sentiment_dictionary()

    # 1. scrape data
    reddit = praw.Reddit(client_id=reddit_client_id, client_secret=reddit_client_secret, user_agent=reddit_user_agent)
    api = PushshiftAPI(reddit)

    # convert calendar date to eastern time
    raw_date = dateutil.parser.parse(str(calendar_date))
    end_unaware = dt.datetime(raw_date.year, raw_date.month, raw_date.day, 5)  # pulls until 5AM NY Time
    timezone = pytz.timezone("America/New_York")
    end_date = timezone.localize(end_unaware)

    if exclude_weekends:
        raw_start = np.busday_offset(end_date.date(), -1).astype(dt.datetime)
        start_unaware = dt.datetime(raw_start.year, raw_start.month, raw_start.day, 5)  # pulls from 5AM NY Time
        start_date = timezone.localize(start_unaware)
    else:
        start_date = end_date - dt.timedelta(hours=24)

    start_epoch = int(start_date.timestamp())
    end_epoch = int(end_date.timestamp())
    posts = list(api.search_submissions(subreddit='wallstreetbets', after=start_epoch, before=end_epoch))
    print(dt.datetime.now(), "->", calendar_date, "Total Posts:", len(posts))

    # limit to top 95% of posts
    posts.sort(key=lambda t: t.score, reverse=True)

    # find total score
    tscore = 0
    for x in range(0, len(posts)):
        tscore += posts[x].score

    # identify post coverage -> we only keep the top 95% of posts
    if tscore > 0:
        cscore = 0
        x = 0
        for x in range(0, len(posts)):
            cscore += posts[x].score
            if cscore / tscore >= 0.95:
                break
        posts = posts[0:x]
        print(dt.datetime.now(), "->", calendar_date, "Kept Posts:", len(posts))

    # 2. match text to tickers, to identify a topic.  often, comments will be related to a master post's subject.
    #   note: multiple subjects are possible, in which case we'll need to use a probabilistic assignment by number
    #   of mentions
    print(dt.datetime.now(), "->", calendar_date, "Processing posts.")
    pool = NestedPool(mp.cpu_count() - num_reserved_cores)
    pool_results = {}
    for post in posts:
        pool_results[post.id] = pool.apply_async(process_post, args=[post, ticker_set, exclude_set, sentiment_dict])
    pool.close()
    pool.join()

    print(dt.datetime.now(), "->", calendar_date, "Counting sentiment totals.")
    total_sentiment_count = 0
    sentiments = {}  # symbol, bull count, bear count
    for post_id in pool_results:
        raw_sentiment = pool_results[post_id].get()
        try:
            this_sentiment = dict(raw_sentiment)  # returns a dictionary
            for ticker in this_sentiment:
                total_sentiment_count += this_sentiment[ticker][0] + this_sentiment[ticker][1]
                if ticker not in sentiments:
                    sentiments[ticker] = this_sentiment[ticker]
                else:
                    sentiments[ticker] = (sentiments[ticker][0] + this_sentiment[ticker][0],
                                          sentiments[ticker][1] + this_sentiment[ticker][1])
        except Exception as e:
            print("Exception for post %s. Error: %s, Raw Result: %s", post_id, e, raw_sentiment)

    pool.terminate()
    del pool

    # now, given the sentiments, go through and keep only results with more than 2% of the total vote, so the
    # max for any given day will be 50 symbols.  this function will also convert the formatting from tuples-based
    # to dictionary-based, and will compute performances
    print(dt.datetime.now(), "->", calendar_date, "Computing performance history.")
    sorted_sentiments = {k: v for k, v in sorted(sentiments.items(), key=lambda item: -(item[1][0] + item[1][1]))}
    hist_begin_date = dateutil.parser.parse(str(calendar_date)) - dt.timedelta(weeks=13)
    spy_hist = get_history("SPY", hist_begin_date, calendar_date)
    vix_hist = get_history("VXX", hist_begin_date, calendar_date)

    sent_dict = {}
    for k in sorted_sentiments:
        vote_share = (sorted_sentiments[k][0] + sorted_sentiments[k][1]) / total_sentiment_count
        unsigned_vote = (sorted_sentiments[k][0] + sorted_sentiments[k][1])
        polarity = ((sorted_sentiments[k][0] - sorted_sentiments[k][1]) / unsigned_vote) if unsigned_vote > 0 else 0

        if vote_share >= min_vote_share:
            # try to get forward history if possible so we can do performance comparison.  only useful when backfilling.
            hist_end_date = dateutil.parser.parse(str(calendar_date)) + dt.timedelta(weeks=5)
            if dt.datetime.now() < hist_end_date:
                hist_end_date = dt.datetime.now()

            try:
                this_hist = get_history(k, hist_begin_date, hist_end_date)
                avg_volume = this_hist[:str(calendar_date)].volume.rolling(60).mean()[-1]
                if avg_volume >= min_avg_volume:
                    sent_dict[k] = dict(positive=sorted_sentiments[k][0], negative=sorted_sentiments[k][1],
                                        vote_share=vote_share, polarity=polarity)
                    sent_dict[k].update(populate_performance(calendar_date, exclude_weekends, polarity,
                                                             this_hist, spy_hist, vix_hist))
            except Exception as e:
                print("Error when processing returns for", k, hist_end_date, ":", e)
        else:
            break

    # strip any sentiments where a direction is not defined
    for k in list(sent_dict.keys()):
        if sent_dict[k]['direction'] is None:
            sent_dict.pop(k)

    sdb.store_sentiments(sent_dict, calendar_date, db_path)
    pstop = dt.datetime.now()
    print(pstop, "Processed:", calendar_date, "Total Time:", pstop - pstart)
    return sent_dict


def backfill_sentiments(backfill_date="2015-01-01", exclude_weekends=True):

    today = dt.datetime.now()
    timezone = pytz.timezone('America/New_York')
    today = timezone.localize(today)
    if today.hour >= 5:
        end_date = today
    else:
        end_date = today - dt.timedelta(days=1)
    end_date = end_date.replace(tzinfo=None)

    if exclude_weekends:
        active_span = pd.bdate_range(backfill_date, end_date)
    else:
        active_span = pd.date_range(backfill_date, end_date)
    active_span = active_span.sort_values(ascending=False)

    # run outside of multiprocessed functions to prevent racing when multiprocessing
    get_ticker_set()
    if not os.path.exists(db_path):
        sdb.initialize_db(db_path)

    for tdate in active_span:
        process_calendar_date(str(tdate), exclude_weekends)

    # do clean-up
    if os.path.exists('Engine/Inputs/ticker_list.csv'):
        os.remove('Engine/Inputs/ticker_list.csv')
    return


def backfill_performance(backfill_date=None, exclude_weekends=True):
    unfilled_sentiments = sdb.fetch_all_sentiments(db_path, backfill_date, True)

    # determine history start date (we'll request 20 days offset from this)
    request_dates = {}
    for this_date in unfilled_sentiments:
        stocks = unfilled_sentiments[this_date]
        for stock in stocks:
            if stock not in request_dates:
                request_dates[stock] = dateutil.parser.parse(this_date)
            else:
                request_dates[stock] = min(dateutil.parser.parse(this_date), request_dates[stock])

    calendar_date = dt.datetime.now().date()
    histories = {}
    for stock in request_dates:
        try:
            base_date = str(request_dates[stock])
            hist_begin_date = dateutil.parser.parse(base_date) - dt.timedelta(weeks=13)
            if 'SPY' not in histories:
                histories['SPY'] = get_history('SPY', hist_begin_date, calendar_date)
            if 'VXX' not in histories:
                histories['VXX'] = get_history('VXX', hist_begin_date, calendar_date)
            histories[stock] = get_history(stock, hist_begin_date, calendar_date)
        except Exception as e:
            print("History fetching error:", e)

    # fill performances for every entry in the unfilled sentiments
    for this_date in unfilled_sentiments:
        stocks_data = unfilled_sentiments[this_date]
        for stock in stocks_data:
            stocks_data[stock].update(
                populate_performance(this_date, exclude_weekends, stocks_data[stock]['polarity'],
                                     histories[stock], histories['SPY'], histories['VXX']))
        sdb.store_sentiments(stocks_data, this_date, db_path)
    return


def daily_maintenance():
    """
    This performs daily maintenance on our database - it scrapes the current day and then goes through any missing
    day's performances to determine how they did.
    :return:
    """

    calendar_date = str(dt.datetime.now().date())
    process_calendar_date(calendar_date)
    backfill_performance(None)
    return
