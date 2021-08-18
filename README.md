#BASED ON tese articles:
# https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
# https://towardsdatascience.com/algorithmic-trading-with-rsi-using-python-f9823e550fe0
# https://medium.com/codex/algorithmic-trading-with-relative-strength-index-in-python-d969cf22dd85

#!!!ta-lib instll and build is tied to Microsoft Visual C++
#so 'talib' it will not be used

#pip install pandas-ta
import pandas_ta as pta

# pandas_ta requires numpy version 1.20 or higher
# pip show numpy
# pip uninstall numpy
# pip install numpy

import numpy as np
import yfinance as yf
import pandas_datareader as web
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
import requests


import pandas

class YFStock():
    def __init__(self, ticker):
        self.ticker = ticker
        #self.predictor = StockPricePredictor(self.ticker)
        #self.news = NewsAnalyzer_YahooFinance()

    def current_price(self):
        stock = yf.Ticker(self.ticker)
        return stock.info["currentPrice"]

    def build_url(self, ticker, start_date, end_date, interval="1d"):
        end_seconds = int(pd.Timestamp(end_date).timestamp())
        start_seconds = int(pd.Timestamp(start_date).timestamp())

        site = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker
        params = {"period1": start_seconds, "period2": end_seconds,
                  "interval": interval.lower(), "events": "div,splits"}

        return site, params

    def download_historical_prices(self, ticker, start_date=None, end_date=None, index_as_date=True, interval="1m"):
        '''Downloads historical stock price data into a pandas data frame.  Interval
           must be "1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data.
           Intraday minute data is limited to 7 days.
           @param: ticker
           @param: start_date = None
           @param: end_date = None
           @param: index_as_date = True
           @param: interval = "1d"
        '''

        if interval not in ("1d", "1wk", "1mo", "1m"):
            raise AssertionError("interval must be of of '1d', '1wk', '1mo', or '1m'")

        # build and connect to URL
        site, params = self.build_url(ticker, start_date, end_date, interval)
        headers = {'user-agent': 'my-agent/1.0.1'}
        resp = requests.get(site, params=params, headers=headers)

        if not resp.ok:
            raise AssertionError(resp.json())

        # get JSON response
        data = resp.json()
        # get open / high / low / close data
        frame = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])
        # get the date info
        temp_time = data["chart"]["result"][0]["timestamp"]

        if interval != "1m":
            # add in adjclose
            frame["adjclose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
            frame.index = pd.to_datetime(temp_time, unit="s")
            frame.index = frame.index.map(lambda dt: dt.floor("d"))
            frame = frame[["open", "high", "low", "close", "adjclose", "volume"]]
        else:
            frame.index = pd.to_datetime(temp_time, unit="s")
            frame = frame[["open", "high", "low", "close", "volume"]]

        frame['ticker'] = ticker.upper()
        if not index_as_date:
            frame = frame.reset_index()
            frame.rename(columns={"index": "date"}, inplace=True)
        return frame

    def get_historical_prices_min(self, hours_back=24):
        dt_now = dt.datetime.now()
        data_frame = self.download_historical_prices(ticker=self.ticker, start_date=(dt_now - dt.timedelta(minutes=hours_back*60)), end_date=dt_now, interval="1m")
        return data_frame

def rsi(df, periods=14, ema=False):
    """
    Returns a pd.Series with the relative strength index.
    """
#    close_delta = df['close'].diff()
    close_delta = df.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema is True:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()


        #tema = 3xEMA - 3*EMA(EMA)+EMA(EMA(EMA))
        two_ma_up = ma_up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        three_ma_up = two_ma_up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_up = 3*ma_up - 3*two_ma_up + three_ma_up

        two_ma_down = ma_down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        three_ma_down = two_ma_down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = 3*ma_down - 3*two_ma_down + three_ma_down

    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods, min_periods=periods).mean()
        ma_down = down.rolling(window=periods, min_periods=periods).mean()

    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

'''
Basic strategy for RSI trading:  when the value leaves the overbought and oversold sections, it makes the appropriate trade. 
For example, if it leaves the oversold section, a buy trade is made. If it leaves the overbought section, a sell trade is made.
'''
def strategy_test(prices, rsi, rsi_length):
    buy_signals = [np.nan]*len(prices)
    sell_signals = [np.nan]*len(prices)
    profit = 0.0
    for i in range(rsi_length+1,len(prices)):
        if rsi[i - 1] <= 30 and rsi[i] > 30:
            buy_signals[i]=prices[i]
        elif rsi[i - 1] >= 70 and rsi[i] < 70:
            sell_signals[i]=prices[i]
    return buy_signals, sell_signals
    #print(f"profit: {profit}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    company = 'GBTC'
    company = 'ZIM'
    rsi_length = 14
    rsi_stride_min = 25
    test_days = 7

    '''
    #DAY data read
    test_end = dt.datetime.now()
    test_start = test_end - dt.timedelta(days=test_days+rsi_length)
    test_data = web.DataReader(company, "yahoo", test_start, test_end)
    #or using yfinance
    #test_data = yf.download(company, test_start, test_end)
    column = "Close"
    '''

    #Minute data read
    stock = YFStock(company)
    test_data = stock.get_historical_prices_min(hours_back=24 * test_days)
    #columns: open/close/high/low/volume
    column = "close"
    #get only every Xth minute
    test_data = test_data[::rsi_stride_min]

    #CALCULATE RSI for each data point (starting from rsi_length-th)
    # first rsi_length RSI values will have numpy.nan value as there as not enough information for consistent RSI calculation
    # https://stackoverflow.com/questions/17534106/what-is-the-difference-between-nan-and-none
    #rsi_ema = pta.rsi(test_data[column], length=rsi_length, ema=True)
    rsi_ema = rsi(df=test_data[column], periods=rsi_length, ema=True)
    rsi_sma = rsi(df=test_data[column], periods=rsi_length, ema=False)

    #buy_signals, sell_signals = strategy_test(test_data[column], rsi_sma, rsi_length)
    buy_signals, sell_signals = strategy_test(test_data[column], rsi_ema, rsi_length)

    fig = plt.figure()
    plt.plot(test_data.index, [70] * len(test_data.index), color="darkgrey", label="overbought")
    plt.plot(test_data.index, [30] * len(test_data.index), color="lightgray", label="oversold")
    plt.plot(test_data.index, rsi_ema, color="wheat", label="rsi ema")
    plt.plot(test_data.index, rsi_sma, color="goldenrod", label="rsi sma")
    plt.plot(test_data[column], color="sienna", label=f"{company} close price")
    plt.scatter(x=test_data.index, y=buy_signals, s=25, color='green', alpha=0.7,
                label=f"Buy signal")
    plt.scatter(x=test_data.index, y=sell_signals, s=25, color='red', alpha=0.7,
                label=f"Sell signal")
    plt.legend()
    plt.show()

    print("done")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
