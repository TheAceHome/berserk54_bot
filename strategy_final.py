import pandas as pd
import numpy as np
import yfinance
from backtesting import Backtest, Strategy
import dataframe_image as dfi
from datetime import datetime


def SMA(values, n):
    return pd.Series(values).rolling(n).mean()


def aroon(Close, n=14):
    df = pd.DataFrame(Close)
    df['up'] = 100 * df['Close'].rolling(window=n + 1, center=False).apply(lambda x: x.argmax()) / n
    df['down'] = 100 * df['Close'].rolling(window=n + 1, center=False).apply(lambda x: x.argmin()) / n
    return (pd.Series(df['up']), pd.Series(df['down']))


def parabolic_sar(new):
    initial_af = 0.02
    step_af = 0.02
    end_af = 0.2
    new['trend'] = 0
    new['sar'] = 0.0
    new['real sar'] = 0.0
    new['ep'] = 0.0
    new['af'] = 0.0

    new['trend'][1] = 1 if new['Close'][1] > new['Close'][0] else -1
    new['sar'][1] = new['High'][0] if new['trend'][1] > 0 else new['Low'][0]
    new.at[1, 'real sar'] = new['sar'][1]
    new['ep'][1] = new['High'][1] if new['trend'][1] > 0 else new['Low'][1]
    new['af'][1] = initial_af

    # calculation
    for i in range(2, len(new)):

        temp = new['sar'][i - 1] + new['af'][i - 1] * (new['ep'][i - 1] - new['sar'][i - 1])
        if new['trend'][i - 1] < 0:
            new.at[i, 'sar'] = max(temp, new['High'][i - 1], new['High'][i - 2])
            temp = 1 if new['sar'][i] < new['High'][i] else new['trend'][i - 1] - 1
        else:
            new.at[i, 'sar'] = min(temp, new['Low'][i - 1], new['Low'][i - 2])
            temp = -1 if new['sar'][i] > new['Low'][i] else new['trend'][i - 1] + 1
        new.at[i, 'trend'] = temp

        if new['trend'][i] < 0:
            temp = min(new['Low'][i], new['ep'][i - 1]) if new['trend'][i] != -1 else new['Low'][i]
        else:
            temp = max(new['High'][i], new['ep'][i - 1]) if new['trend'][i] != 1 else new['High'][i]
        new.at[i, 'ep'] = temp

        if np.abs(new['trend'][i]) == 1:
            temp = new['ep'][i - 1]
            new.at[i, 'af'] = initial_af
        else:
            temp = new['sar'][i]
            if new['ep'][i] == new['ep'][i - 1]:
                new.at[i, 'af'] = new['af'][i - 1]
            else:
                new.at[i, 'af'] = min(end_af, new['af'][i - 1] + step_af)
        new.at[i, 'real sar'] = temp
    return new


def signal_generation(method, Open, High, Low, Close):
    df = pd.DataFrame(Open)
    df['High'] = High
    df['Low'] = Low
    df['Close'] = Close
    new = method(df)

    new['positions'], new['signals'] = 0, 0
    new['positions'] = np.where(new['real sar'] < new['Close'], 1, 0)
    new['signals'] = new['positions'].diff()

    return pd.Series(new['signals'])


def MACD(Close, n, m):
    df = pd.DataFrame(Close)
    df['mav_5'] = df['Close'].ewm(span=n, adjust=False).mean()
    df['mav_10'] = df['Close'].ewm(span=m, adjust=False).mean()
    df['macd'] = df['mav_5'] - df['mav_10']
    df['exp3'] = df['macd'].ewm(span=20, adjust=False).mean()
    return (pd.Series(df['exp3']), pd.Series(df['macd']))


class Strategy_MACD(Strategy):
    n = 12
    m = 26

    def init(self):

        self.cr1, self.cr2 = self.I(MACD, self.data.Close, self.n, self.m)

    def next(self):

        if (self.cr1[-2] < self.cr2[-2] and
                self.cr1[-1] > self.cr2[-1]):
            self.sell()

        elif (self.cr1[-2] > self.cr2[-2] and
              self.cr1[-1] < self.cr2[-1]):
            self.buy()


class Strategy_SAR(Strategy):

    def init(self):

        self.sar1 = self.I(signal_generation, parabolic_sar, self.data.Open, self.data.High, self.data.Low,
                           self.data.Close)

    def next(self):

        if self.sar1 == 1:
            self.buy()
        elif self.sar1 == -1:
            self.sell()


class Strategy_SMA(Strategy):
    n = 20
    m = 14

    def init(self):
        self.sma30 = self.I(SMA, self.data.Close, self.n)
        self.sma60 = self.I(SMA, self.data.Close, self.m)
        self.s = True

    def next(self):

        if self.sma30 > self.sma60 and self.s == False:
            self.sell()
            self.s = True
        elif self.sma30 < self.sma60 and self.s == True:
            self.buy()
            self.s = False


def try_strategy(ticker, start_date, end_date):
    tickers = list(ticker.split(","))
    print(tickers)
    result = pd.DataFrame()
    for i in tickers:
        df = yfinance.download(i, start=start_date, end=end_date, progress=False)
        bt = Backtest(df, Strategy_SMA, commission=.005,
                      exclusive_orders=True, cash=1000000.0)
        stats, heatmap = bt.optimize(
            n=[9, 14, 20, 35],
            m=[9, 14, 20, 35],
            maximize='Return [%]',
            max_tries=200,
            random_state=0,
            return_heatmap=True)
        result[i] = stats
    dfi.export(result, 'graphs/result.png')


def comp_strategy():
    df = pd.read_csv('database.csv', header=0, sep=';')
    date=datetime.today().strftime('%Y-%m-%d')
    actions_all = []
    for i in range(len(df['nickname'])):
        tickers = df['tickers'][i].split(",")
        sma = df['sma'][i].split(",")
        sma2=[int(i) for i in sma]
        actions=[]
        for j in tickers:
            try:
                data = yfinance.download(j, start='2020-01-01', end=date, progress=False)
                data['sma10'] = SMA(data['Close'], sma2[0])
                data['sma20'] = SMA(data['Close'], sma2[1])
                s = True
                list = []
                for i in range(len( data['sma10'])):
                    if  data['sma10'][i] >  data['sma20'][i] and s == False:
                        list.append('sell')
                        s = True
                    elif  data['sma10'][i] <  data['sma20'][i] and s == True:
                        list.append('buy')
                        s = False
                    else:
                        list.append('wait')
                actions.append(list[-1])
            except:
                pass
        actions_all.append(actions)
    df['actions_all']=actions_all
    df.to_csv('database.csv', sep=';', index=False)

