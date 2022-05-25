# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:55:56 2022

@author: BruceKing
"""

import pandas as pd
from cryptocmd import CmcScraper
import pickle
def get_crypto_data(coin_name):
    scraper = CmcScraper(coin_name)
    coin_df = scraper.get_dataframe()
    coin_df = pd.DataFrame(coin_df, columns=['Date','Close'])
    coin_df.set_index('Date', inplace=True)
    coin_df = coin_df.rename(columns = {'Close':coin_name})
    coin_df = coin_df.iloc[::-1]
    with open(coin_name + "_data", "wb") as fp:   #Pickling
        pickle.dump(coin_df, fp)


