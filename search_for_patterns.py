from scipy.signal import argrelextrema
from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpl


def find_extrema(s, bw='cv_ls'):

    # Copy series so we can replace index and perform non-parametric
    # kernel regression.
    prices = s.copy()
    prices = prices.reset_index()
    prices.columns = ['date', 'price']
    prices = prices['price']

    kr = KernelReg([prices.values], [prices.index.to_numpy()], var_type='c', bw=bw)
    f = kr.fit([prices.index])

    # Use smoothed prices to determine local minima and maxima
    smooth_prices = pd.Series(data=f[0], index=prices.index)
    smooth_local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    smooth_local_min = argrelextrema(smooth_prices.values, np.less)[0]
    local_max_min = np.sort(np.concatenate([smooth_local_max, smooth_local_min]))
    smooth_extrema = smooth_prices.loc[local_max_min]

    # Iterate over extrema arrays returning datetime of passed
    # prices array. Uses idxmax and idxmin to window for local extrema.
    price_local_max_dt = []
    for i in smooth_local_max:
        if (i > 1) and (i < len(prices) - 1):
            price_local_max_dt.append(prices.iloc[i - 2:i + 2].idxmax())

    price_local_min_dt = []
    for i in smooth_local_min:
        if (i > 1) and (i < len(prices) - 1):
            price_local_min_dt.append(prices.iloc[i - 2:i + 2].idxmin())

    maxima = pd.Series(prices.loc[price_local_max_dt])
    minima = pd.Series(prices.loc[price_local_min_dt])
    extrema = pd.concat([maxima, minima]).sort_index()

    # Return series for each with bar as index
    return extrema, prices, smooth_extrema, smooth_prices


from collections import defaultdict


def find_patterns(extrema, max_bars=35):
    patterns = defaultdict(list)

    # Need to start at five extrema for pattern generation
    for i in range(5, len(extrema)):
        window = extrema.iloc[i - 5:i]

        # A pattern must play out within max_bars (default 35)
        if (window.index[-1] - window.index[0]) > max_bars:
            continue

        # Using the notation from the paper to avoid mistakes
        e1 = window.iloc[0]
        e2 = window.iloc[1]
        e3 = window.iloc[2]
        e4 = window.iloc[3]
        e5 = window.iloc[4]

        rtop_g1 = np.mean([e1, e3, e5])
        rtop_g2 = np.mean([e2, e4])

        # Head and Shoulders
        if (e1 > e2) and (e3 > e1) and (e3 > e5) and \
                (abs(e1 - e5) <= 0.03 * np.mean([e1, e5])) and \
                (abs(e2 - e4) <= 0.03 * np.mean([e1, e5])):
            patterns['HS'].append((window.index[0], window.index[-1]))

        # Inverse Head and Shoulders
        elif (e1 < e2) and (e3 < e1) and (e3 < e5) and \
                (abs(e1 - e5) <= 0.03 * np.mean([e1, e5])) and \
                (abs(e2 - e4) <= 0.03 * np.mean([e1, e5])):
            patterns['IHS'].append((window.index[0], window.index[-1]))

        # Broadening Top
        elif (e1 > e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['BTOP'].append((window.index[0], window.index[-1]))

        # Broadening Bottom
        elif (e1 < e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['BBOT'].append((window.index[0], window.index[-1]))

        # Triangle Top
        elif (e1 > e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['TTOP'].append((window.index[0], window.index[-1]))

        # Triangle Bottom
        elif (e1 < e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['TBOT'].append((window.index[0], window.index[-1]))

        # Rectangle Top/бычий прямоугольник
        elif (e1 > e2) and (abs(e1 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e3 - rtop_g1) / rtop_g1 < 0.0075) and (abs(e5 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e2 - rtop_g2) / rtop_g2 < 0.0075) and (abs(e4 - rtop_g2) / rtop_g2 < 0.0075) and \
                (min(e1, e3, e5) > max(e2, e4)):
            patterns['RTOP'].append((window.index[0], window.index[-1]))

        # Rectangle Bottom/медвежий прямоугольник
        elif (e1 < e2) and (abs(e1 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e3 - rtop_g1) / rtop_g1 < 0.0075) and (abs(e5 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e2 - rtop_g2) / rtop_g2 < 0.0075) and (abs(e4 - rtop_g2) / rtop_g2 < 0.0075) and \
                (max(e1, e3, e5) > min(e2, e4)):
            patterns['RBOT'].append((window.index[0], window.index[-1]))

    return patterns


def Read_csv(ticker, start, stop):
    import yfinance
    df = yfinance.download(ticker, progress=False, start=start, end=stop)
    return df


def signal_to_start(close, tbot):
    signal = []
    sign = []
    k = 0
    for i, j in tbot:
        sign.append(close.iloc[[i]][0])
    for i in close:
        if i in sign:
            signal.append(close.iloc[[k]][0] * 0.99)
        else:
            signal.append(np.nan)
        k += 1
    return signal


def signal_to_end(close, tbot):
    signal2 = []
    sign = []
    k = 0
    for i, j in tbot:
        sign.append(close.iloc[[j]][0])
    for i in close:
        if i in sign:
            signal2.append(close.iloc[[k]][0] * 1.01)
        else:
            signal2.append(np.nan)
        k += 1
    return signal2


def find_pattern(ticker, start, stop, pattern):
    df = Read_csv(ticker, start, stop)
    extrema, prices, smooth_extrema, smooth_prices = find_extrema(df['Close'], bw=[1.5])
    patterns = find_patterns(extrema)
    print(patterns['HS'])
    for name, pattern_periods in patterns.items():
        print(f"{name}: {len(pattern_periods)} найдено")
    apds = [mpl.make_addplot(signal_to_start(df['Low'], patterns[str(pattern)]), type='scatter', marker='^'),
            mpl.make_addplot(signal_to_end(df['High'], patterns[str(pattern)]), type='scatter', marker='v')]
    fig, axes = mpl.plot(df, addplot=apds, type='candle', returnfig=True)
    axes[0].set_title(ticker)
    fig.savefig('graphs/patterns.png')


def get_text_for_graph(message):
    txt = message.text
    txt = txt.split(' ')
    return find_pattern(txt[1], txt[2], txt[3], txt[4])
