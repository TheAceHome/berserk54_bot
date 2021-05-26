import pandas as pd


def check_name_in_csv(nick):
    df = pd.read_csv('database.csv', header=0, sep=';')
    if nick in df['nickname'].values:
        print('Nickname already existing')
        return True


def add_nick_to_df(nick):
    if check_name_in_csv(nick):
        return None
    else:
        df = pd.read_csv('database.csv', header=0, sep=';')
        df = df.append({'nickname': nick, 'tickers': 1}, ignore_index=True)
        df.to_csv('database.csv', sep=';', index=False)
        print('Nickname added')


def set_tickers(nick,tickers,sma):
    df = pd.read_csv('database.csv', header=0, sep=';', index_col=0)
    add_nick_to_df(nick)
    df.loc[nick,'tickers'] = tickers
    df.loc[nick,'sma'] = sma
    df.to_csv('database.csv', sep=';')

