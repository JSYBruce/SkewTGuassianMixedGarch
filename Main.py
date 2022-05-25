# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:59:27 2022

@author: BruceKing
"""
from DoubleNormal import DoubleNormal
from Model import ARCHModel
from Garch import GARCH
from ArchResult import ARCHModelResult
from arch import arch_model
from DataScraper import get_crypto_data
import pickle
import numpy as np

def runModel(Coinname):
    try:
        with open(Coinname + '_data', 'rb') as f:
            coin_df = pickle.load(f)
    except Exception:
        get_crypto_data(Coinname)
        with open(Coinname + '_data', 'rb') as f:
            coin_df = pickle.load(f)
        pass
    with open(Coinname + '_data', 'rb') as f:
        coin_df = pickle.load(f)
    coin_df['returns'] = np.log(coin_df[Coinname]).diff().mul(100).dropna()
    coin_df.dropna(inplace=True)
    date = coin_df.index[1:]
    np.savetxt("time.csv",np.array(date), delimiter = ",")
    coin_df['returns'] = np.log(coin_df[Coinname]).diff().mul(100).dropna()
    coin_df.dropna(inplace=True)
    archmodel = ARCHModel(y = coin_df.returns, distribution = DoubleNormal(), volatility = GARCH())
    result = archmodel.fit(starting_values=[-2,2, 0.23763582, 0.2    ,0.2    ,0.9   ,0.5 ,0.4, 5,0.5])
    result = ARCHModelResult(result[0], None,  result[1],  result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12])
    return result
    
def runModelSingle(Coinname, Model):
    with open(Coinname + '_data', 'rb') as f:
        coin_df = pickle.load(f)
    coin_df['returns'] = np.log(coin_df[Coinname]).diff().mul(100).dropna()
    coin_df.dropna(inplace=True)
    gjrgarch_gm = arch_model(coin_df.returns, p=1, q=1, mean='constant', vol='GARCH', dist = Model) 
    result = gjrgarch_gm.fit()
    print(result.summary())
