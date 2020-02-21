from utilities.sequencer import Sequencer
from utilities.dataframe_utils import read_db
from utilities.dataframe_utils import write_blocks_on_disk
from utilities.dataframe_utils import generate_blocks_from_lists
from utilities.dataframe_utils import generate_stations_dictionaries
from utilities.sequence_dataloader import SequenceDataLoader_1h_int_3h_seq
from utilities.config import init_args

from time import perf_counter
from time import sleep

import pandas as pd
import numpy as np
import tensorflow as tf

DEBUG = 1    # To set to 0 to test with real crops
ROOT_DIR = '.'
CLEANDF_PATH = ROOT_DIR + '/data/clean_df'



print('Step 1: Reading clean dataframe...')
clean_df = pd.read_pickle("./clean_df")


print('Step 2: Generating records...')
stations_list = np.unique(clean_df['station'])
records = generate_stations_dictionaries(clean_df, stations_list)

print('Step 3: Generating the joint table (previous steps could be skipped if the joint table is already there)...')
db = write_blocks_on_disk(records, stations_list, root_dir='./data/preprocessed/')
#db = pd.read_pickle('database.db')

print('Step 4: Generating the mappings for the sequencer...')
stations_mappings = read_db(db)
stations_names = np.unique(clean_df['station']).tolist()

print('Step 5: Instantiating the sequencer...')
sequencer = Sequencer(stations_list, stations_mappings, offset=1800, seq_length=16, max_batch_size = 50)


print('Step 6: Loading data...')
tic = perf_counter()

toc = perf_counter()

print('Total time elapsed during loading of the data: %f' %(toc - tic))