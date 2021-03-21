import emoji
import random
from termcolor import colored

import ta
import datetime
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
    load_dotenv()
    CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')
    if not CRYPTOCOMPARE_API_KEY:
        raise EnvironmentError('CRYPTOCOMPARE_API_KEY not found on .env')
    df_train_path = 'data/bot_train_{}_{}_{}.csv'.format(
        asset + currency, datapoints, granularity)
    df_rollout_path = 'data/bot_rollout_{}_{}_{}.csv'.format(
        asset + currency, datapoints, granularity)
    if not os.path.exists(df_rollout_path):
        headers = {'User-Agent': 'Mozilla/5.0',
                   'authorization': 'Apikey {}'.format(CRYPTOCOMPARE_API_KEY)}

        url = 'https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym={}&limit={}&e={}'.format(
            granularity, asset, currency, datapoints, exchange)
        with yaspin(text='Downloading datasets') as sp:
            print(url)
            response = requests.get(url, headers=headers)
            sp.hide()
            print_formatted_text(HTML(
                u'<b>></b> <msg>{}/{}</msg> <sub-msg>download complete</sub-msg>'.format(
                    asset, currency)
            ), style=style)
            sp.show()

        json_response = response.json()
        status = json_response['Response']
        if status == "Error":
            raise AssertionError(colored(json_response['Message'], 'red'))
        result = json_response['Data']
        data = pd.DataFrame(result)
        print("--------1")

        df = pd.DataFrame()
        df['Date'] = pd.to_datetime(data['time'], utc=True, unit='s')
        df['open'] = data['open']
        df['high'] = data['high']
        df['low'] = data['low']
        df['close'] = data['close']
        df['volumefrom'] = data['volumefrom']
        print("--------2")
        print(df.tail())
        # indicators
        # https://github.com/mrjbq7/ta-lib/blob/master/docs/func.md
        print("--------2")
        features = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volumefrom', fillna=True)
        print(features.head(3))
        features.drop(columns=['open', 'high', 'low', 'close', 'volumefrom'])

        # df.fillna(df.mean(), inplace=True)
        features.dropna(inplace=True)
        features.set_index('Date', inplace=True)
        # print(colored('> caching' + asset + '/' + currency + ':)', 'cyan'))
        # 75% to train -> test with different value
        train_size = round(len(features) * df_train_size)
        df_train = features[:train_size]
        df_rollout = features[train_size:]
        df_train.to_csv(df_train_path)
        df_rollout.to_csv(df_rollout_path)
        # re-read to avoid indexing issue w/ Ray
        df_train = pd.read_csv(df_train_path)
        df_rollout = pd.read_csv(df_rollout_path)
    else:

        print_formatted_text(HTML(
            u'<b>></b> <msg>{}/{}</msg> <sub-msg>cached</sub-msg>'.format(
                asset, currency)
        ), style=style)

        # print(colored('> feching ' + asset + '/' + currency + ' from cache :)', 'magenta'))
        df_train = pd.read_csv(df_train_path)
        df_rollout = pd.read_csv(df_rollout_path)
        # df_train.set_index('Date', inplace=True)
        # df_rollout.set_index('Date', inplace=True)

    return df_train, df_rollout
