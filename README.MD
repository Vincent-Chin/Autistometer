This project relies on the Natural Language Toolkit for Python at https://www.nltk.org/.  You will need to install this
in your environment before the project will run.  There is a convenience function in main::do_nltk_installations() that
should take care of this for you.

Also note that this project relies on Tiingo at https://www.tiingo.com/ for the historical data used to analyze
holding performance.  You will need to obtain an API key and put it into the credentials.py for the project to work.

You will also need a reddit API key so that you can programmatically scrape reddit, also placed in credentials.py.

-----

To run this project, you normally would schedule a CRON task to run the main::daily_maintenance() function daily.  An
example script exists in Automation/synchronize.sh (for Linux environments.)

However, you can analyze a specific date by calling main::process_calendar_date(), which scours posts on a given date
for sentiment, and then separately you can call backfill_performance to figure out the performance of the identified
tickers.

The main.py code will look at the global variable db_path to inform where to look for the database.  It will create a
fresh database if one does not exist.

-----

There is some partially started visualization code in the Web directory, but it is non-functional as of now.