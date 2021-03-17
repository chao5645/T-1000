import emoji
import random
from termcolor import colored

import datetime
import talib
import os
import colorama
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from yaspin import yaspin
from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.styles import Style

# build a basic prompt_toolkit style for styling the HTML wrapped text
style = Style.from_dict({
    'msg': '#4caf50 bold',
    'sub-msg': '#616161 italic',
    'loading': '#c9c344 italic'
})

colorama.init()


def loading():
    emojis = [':moneybag:', ':yen:', ':dollar:', ':pound:', ':euro:']
    print_formatted_text(
        HTML(u'<b>> {}</b> <loading>loading...</loading>'.format(emoji.emojize(random.choice(emojis), use_aliases=True))), style=style)


def load_csv(filename):
    df = pd.read_csv('data/' + filename, skiprows=1)
    df.drop(columns=['symbol', 'volume_btc'], inplace=True)

    # Fix timestamp form "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    df['date'] = df['date'].str[:14] + '00-00 ' + df['date'].str[-2:]

    # Convert the date column type from string to datetime for proper sorting.
    df['date'] = pd.to_datetime(df['date'])

    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by='date', ascending=True, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')

    return df

def get_datasets(asset, currency, granularity, datapoints, exchange, df_train_size=0.75):
    """Fetch the API and precess the desired pair

    Arguments:
        asset {str} -- First pair
        currency {str} -- Second pair
        granularity {str ['day', 'hour']} -- Granularity
        datapoints {int [100 - 2000]} -- [description]

    Returns:
        pandas.Dataframe -- The OHLCV and indicators dataframe
    """

    df_train_path = 'data/bot_train_{}_{}_{}.csv'.format(
        asset + currency, datapoints, granularity)
    df_rollout_path = 'data/bot_rollout_{}_{}_{}.csv'.format(
        asset + currency, datapoints, granularity)

    df = load_csv("Coinbase_BTCUSD_1h.csv")
    print(df.tail())

    df['Date'] = pd.to_datetime(df['date'])
    df.drop('date', axis=1, inplace=True)

    # indicators
    # https://github.com/mrjbq7/ta-lib/blob/master/docs/func.md
    open_price, high, low, close = np.array(df['open']), np.array(
        df['high']), np.array(df['low']), np.array(df['close'])
    volume = np.array(df['volume'], dtype=float)
    # cycle indicators
    df.loc[:, 'HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df.loc[:, 'HT_DCPHASE'] = talib.HT_DCPHASE(close)
    df.loc[:, 'HT_PHASOR_inphase'], df.loc[:,
                                           'HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    df.loc[:, 'HT_SINE_sine'], df.loc[:,
                                      'HT_SINE_leadsine'] = talib.HT_SINE(close)
    df.loc[:, 'HT_TRENDMODE'] = talib.HT_TRENDMODE(close)
    # momemtum indicators
    df.loc[:, 'ADX'] = talib.ADX(high, low, close, timeperiod=12)
    df.loc[:, 'ADXR'] = talib.ADXR(high, low, close, timeperiod=13)
    df.loc[:, 'APO'] = talib.APO(
        close, fastperiod=5, slowperiod=10, matype=0)
    df.loc[:, 'AROON_down'], df.loc[:, 'AROON_up'] = talib.AROON(
        high, low, timeperiod=15)
    df.loc[:, 'AROONOSC'] = talib.AROONOSC(high, low, timeperiod=13)
    df.loc[:, 'BOP'] = talib.BOP(open_price, high, low, close)
    df.loc[:, 'CCI'] = talib.CCI(high, low, close, timeperiod=13)
    df.loc[:, 'CMO'] = talib.CMO(close, timeperiod=14)
    df.loc[:, 'DX'] = talib.DX(high, low, close, timeperiod=10)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        close, fastperiod=5, slowperiod=10, signalperiod=20)
    df.loc[:, 'MFI'] = talib.MFI(high, low, close, volume, timeperiod=12)
    df.loc[:, 'MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=10)
    df.loc[:, 'MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    df.loc[:, 'MOM'] = talib.MOM(close, timeperiod=20)
    df.loc[:, 'PPO'] = talib.PPO(
        close, fastperiod=17, slowperiod=35, matype=2)
    df.loc[:, 'ROC'] = talib.ROC(close, timeperiod=12)
    df.loc[:, 'RSI'] = talib.RSI(close, timeperiod=25)
    df.loc[:, 'STOCH_k'], df.loc[:, 'STOCH_d'] = talib.STOCH(
        high, low, close, fastk_period=35, slowk_period=12, slowk_matype=0, slowd_period=7, slowd_matype=0)
    df.loc[:, 'STOCHF_k'], df.loc[:, 'STOCHF_d'] = talib.STOCHF(
        high, low, close, fastk_period=28, fastd_period=14, fastd_matype=0)
    df.loc[:, 'STOCHRSI_K'], df.loc[:, 'STOCHRSI_D'] = talib.STOCHRSI(
        close, timeperiod=35, fastk_period=12, fastd_period=10, fastd_matype=1)
    df.loc[:, 'TRIX'] = talib.TRIX(close, timeperiod=30)
    df.loc[:, 'ULTOSC'] = talib.ULTOSC(
        high, low, close, timeperiod1=14, timeperiod2=28, timeperiod3=35)
    df.loc[:, 'WILLR'] = talib.WILLR(high, low, close, timeperiod=35)
    # overlap studies
    df.loc[:, 'BBANDS_upper'], df.loc[:, 'BBANDS_middle'], df.loc[:, 'BBANDS_lower'] = talib.BBANDS(
        close, timeperiod=12, nbdevup=2, nbdevdn=2, matype=0)
    df.loc[:, 'DEMA'] = talib.DEMA(close, timeperiod=30)
    df.loc[:, 'EMA'] = talib.EMA(close, timeperiod=7)
    df.loc[:, 'HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
    df.loc[:, 'KAMA'] = talib.KAMA(close, timeperiod=5)
    df.loc[:, 'MA'] = talib.MA(close, timeperiod=5, matype=0)
    df.loc[:, 'MIDPOINT'] = talib.MIDPOINT(close, timeperiod=20)
    df.loc[:, 'WMA'] = talib.WMA(close, timeperiod=15)
    df.loc[:, 'SMA'] = talib.SMA(close)
    # pattern recoginition
    df.loc[:, 'CDL2CROWS'] = talib.CDL2CROWS(open_price, high, low, close)
    df.loc[:, 'CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(
        open_price, high, low, close)
    df.loc[:, 'CDL3INSIDE'] = talib.CDL3INSIDE(
        open_price, high, low, close)
    df.loc[:, 'CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(
        open_price, high, low, close)
    # price transform
    df.loc[:, 'WCLPRICE'] = talib.WCLPRICE(high, low, close)
    # statistic funcitons
    df.loc[:, 'BETA'] = talib.BETA(high, low, timeperiod=20)
    df.loc[:, 'CORREL'] = talib.CORREL(high, low, timeperiod=20)
    df.loc[:, 'STDDEV'] = talib.STDDEV(close, timeperiod=20, nbdev=1)
    df.loc[:, 'TSF'] = talib.TSF(close, timeperiod=20)
    df.loc[:, 'VAR'] = talib.VAR(close, timeperiod=20, nbdev=1)
    # volatility indicators
    df.loc[:, 'ATR'] = talib.ATR(high, low, close, timeperiod=7)
    df.loc[:, 'NATR'] = talib.NATR(high, low, close, timeperiod=20)
    df.loc[:, 'TRANGE'] = talib.TRANGE(high, low, close)
    # volume indicators
    df.loc[:, 'AD'] = talib.AD(high, low, close, volume)
    df.loc[:, 'ADOSC'] = talib.ADOSC(
        high, low, close, volume, fastperiod=10, slowperiod=20)
    df.loc[:, 'OBV'] = talib.OBV(close, volume)

    # df.fillna(df.mean(), inplace=True)
    df.dropna(inplace=True)
    df.set_index('Date', inplace=True)
    # print(colored('> caching' + asset + '/' + currency + ':)', 'cyan'))
    # 75% to train -> test with different value
    train_size = round(len(df) * df_train_size)
    df_train = df[:train_size]
    df_rollout = df[train_size:]
    df_train.to_csv(df_train_path)
    df_rollout.to_csv(df_rollout_path)
    # re-read to avoid indexing issue w/ Ray
    df_train = pd.read_csv(df_train_path)
    df_rollout = pd.read_csv(df_rollout_path)

    return df_train, df_rollout
