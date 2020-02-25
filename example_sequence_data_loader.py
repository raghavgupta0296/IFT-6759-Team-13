from utilities.sequence_dataloader import preprocess_data
from utilities.sequence_dataloader import SequenceDataLoader_1h_int_6h_seq
from utilities.sequence_dataloader import benchmark

from time import perf_counter

import pandas as pd
import numpy as np

DEBUG = 1    # To set to 0 to test with real crops
ROOT_DIR = '.'
CLEANDF_PATH = ROOT_DIR + '/data/clean_df'


print('Step 1: Reading clean dataframe...')
args = init_args()
clean_df = pd.read_pickle(CLEANDF_PATH)
stations_names = np.unique(clean_df['station'])
"""
print('Step 2: Generating sequencer...')
sequencer = preprocess_data(clean_df, root_dir='./data/preprocessed/', db_path = './data/database.db',
                            from_db = True, offset = 18000, seq_length=17, batch_size = 50)

print('Step 3: Loading data...')
tic = perf_counter()
benchmark(SequenceDataLoader_30min_int_9h_seq(sequencer))
toc = perf_counter()
"""

print('Step 1: Generating sequencer...')
sequencer = preprocess_data(args, clean_df, root_dir='./data/preprocessed/', db_path = './data/database.db',
                            from_db = True, offset = 36000, seq_length=3, batch_size = 50)

print('Step 2: Loading data...')
tic = perf_counter()
benchmark(SequenceDataLoader_1h_int_6h_seq(sequencer))
toc = perf_counter()

print('Total time elapsed during loading of the data: %f' %(toc - tic))