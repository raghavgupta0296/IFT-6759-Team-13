# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:13:30 2020

@author: Yassir
"""

from utilities.config import init_args
from utilities.dataframe_utils import generate_memory_blocks
from time import perf_counter

import pandas as pd
import numpy as np

DEBUG = 1    # To set to 0 to test with real crops
ROOT_DIR = '.'
TRAIN_DF = ROOT_DIR + '/data/train_df'
VALID_DF = ROOT_DIR + '/data/valid_df'
TEST_DF = ROOT_DIR + '/data/test_df'
STATION_IDS = ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU',  'SXF']
LIST_DFS = [TRAIN_DF, VALID_DF, TEST_DF]
LIST_DBS = ['./data/train_database.db', './data/valid_database.db', './data/test_database.db']



print('Starting the counter...')
t_init = perf_counter()

args = init_args()
for i, elem in enumerate(LIST_DFS):
    tic = perf_counter()
    print('Reading dataframe...')
    df = pd.read_pickle(elem)
    print('Generating records and the joint table...')
    db = generate_memory_blocks(args, df, STATION_IDS, root_dir = './data/preprocessed/', db_path = LIST_DBS[i])
    toc = perf_counter()
    print('Time elapsed during this step: %f' %(toc - tic))

t_final = perf_counter()

print('Total time elapsed during preprocessing of the data: %f' %(t_final - t_init))