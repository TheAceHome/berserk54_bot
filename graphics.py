import mplfinance as mpl
import yfinance
import numpy as np


def Read_csv(ticker, start, stop):
    df = yfinance.download(ticker, progress=False, start=start, end=stop)
    return df


def get_text_for_graph(message):
    txt = message.text
    txt = txt.split(' ')
    return plot_japan(txt[1], txt[2], txt[3])


def signal_to_buy(switch, close):
    signal = []
    swi = switch[0]
    data=close.index[0]
    for date, value in switch.items():
        if swi != switch[date]:
            if value:
                signal.append(close[data] * 0.99)
            else:
                signal.append(np.nan)
            swi = value
        else:
            signal.append(np.nan)
        data=date
    return signal


def signal_to_sell(switch, close):
    signal2 = []
    swi = switch[0]
    data=close.index[0]
    for date, value in switch.items():
        if swi != switch[date]:
            if value:
                signal2.append(np.nan)
            else:
                signal2.append(close[data] * 1.01)
            swi = value
        else:
            signal2.append(np.nan)
        data=date
    return signal2


def plot_japan(ticker='AAPL', start='2020-01-01', stop='2021-01-01', graph_type='candle', mav_titles=(5, 10)):
    df = Read_csv(ticker, start, stop)
    fig, axes = mpl.plot(df, type=graph_type, mav=mav_titles, returnfig=True)

    if graph_type == 'line':
        mav_titles = list(mav_titles)
        mav_titles.insert(0, ticker)
    axes[0].legend(mav_titles)
    axes[0].set_title(ticker)
    fig.savefig('graphs/japan.png')


def plot_with_signal(ticker, start, stop, mav_titles=(5, 10)):
    df = Read_csv(ticker, start, stop)
    df['mav_5'] = df['Close'].rolling(window=mav_titles[0]).mean()
    df['mav_10'] = df['Close'].rolling(window=mav_titles[1]).mean()
    df['switch'] = df['mav_5'] > df['mav_10']
    apds = [mpl.make_addplot(signal_to_buy(df['switch'], df['Low']), type='scatter', marker='^'),
            mpl.make_addplot(signal_to_sell(df['switch'], df['High']), type='scatter', marker='v')]
    fig, axes = mpl.plot(df, addplot=apds, mav=mav_titles, returnfig=True)
    axes[0].legend(mav_titles)
    axes[0].set_title(ticker)
    fig.savefig('graphs/plot_with_signal.png')
