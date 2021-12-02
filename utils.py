#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Purpose: some misc utility functions

import importlib.util
import warnings
import os

import pandas as pd
import numpy as np
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)


infile = './2021-04-27-16-03 02.csv'
outfile = './2021-04-27-16-03 02.h5'

def sqlcsv_preprocess(filename):
    def wh_to_xy(x):
        x[2] = x[0] + x[2]
        x[3] = x[1] + x[3]
        if x[0] >= 0 and x[0] < 5:
            x[0] = 3
        return x
    df = pd.read_csv(filename, parse_dates=['time'])
    df['time'] = df['time'].dt.strftime("%Y_%m_%d-%H_%M_%S.jpg")
    for i in range(1, 5):
        df.iloc[:,i] = (df.iloc[:,i]
                        .apply(lambda x:np.array(x.split(','), dtype=int))
                        .apply(wh_to_xy))
    return df

def parse_params(path):
    # load parameters
    spec = importlib.util.spec_from_file_location(
        'params', os.path.join(path, 'params.py'))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    params = loader.params
    return params


if __name__ == '__main__':
    df = sqlcsv_preprocess(infile)
    df.to_hdf(outfile, key='df', mode='w')

    ## To read h5
    #  pd.read_hdf("store_tl.h5", "df")
