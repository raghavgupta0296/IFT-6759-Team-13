from utilities.sequence_dataloader import preprocess_data
from utilities.sequence_dataloader import SequenceDataLoader_1h_int_3h_seq
from utilities.sequence_dataloader import SequenceDataLoader_30min_int_9h_seq
from utilities.sequence_dataloader import benchmark
from utilities.sequencer_utils import print_batch

from time import perf_counter

import pandas as pd
import numpy as np

DEBUG = 1    # To set to 0 to test with real crops
ROOT_DIR = '.'
CLEANDF_PATH = ROOT_DIR + '/data/clean_df'


print('Step 1: Reading clean dataframe...')
clean_df = pd.read_pickle(CLEANDF_PATH)
stations_names = np.unique(clean_df['station'])


print('Step 2: Generating sequencer...')
sequencer = preprocess_data(clean_df, root_dir='./data/preprocessed/', db_path = './data/database.db',
                            from_db = True, offset = 18000, seq_length=17, batch_size = 50)

print('Step 2: Loading data...')
tic = perf_counter()
counter = 0
total_samples = 0
batch = sequencer.generate_batch()
while (batch):
    print_batch(batch , './output/batch_' + str(counter) + '_.txt')
    total_samples += len(batch)
    counter += 1
toc = perf_counter()
print('Total number of samples: %d' %total_samples)
print('Total time elapsed during loading of the data: %f' %(toc - tic))