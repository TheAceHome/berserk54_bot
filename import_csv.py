import pandas as pd
import datetime

df = pd.read_csv('GAZP.csv')

if '<TICKER>' in df.columns:
    df.drop('<TICKER>', inplace=True, axis=1)
if '<PER>' in df.columns:
    df.drop('<PER>', inplace=True, axis=1)
if '<TIME>' in df.columns:
    df.drop('<TIME>', inplace=True, axis=1)

df['<DATE>'] = df['<DATE>'].apply(lambda x: (datetime.datetime.strptime(x, "%m/%d/%y").strftime('%Y-%m-%d')))
df['<DATE>'] = pd.to_datetime(df['<DATE>'])
try:
    df.rename(columns={'<DATE>': "Date", '<OPEN>': "Open", '<HIGH>': "High", '<LOW>': "Low", '<CLOSE>': "Close",
                       '<VOL>': "Vol"}, inplace=True)
except:
    print('Не хватает данных')

try:
    df.set_index('Date', inplace=True)
except:
    print('Нет даты')


