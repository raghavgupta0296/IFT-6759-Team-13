import numpy as np
import pandas as pd
import tensorflow as tf

from utilities.dataframe_utils import write_blocks_on_disk
from utilities.dataframe_utils import generate_stations_dictionaries
from utilities.dataframe_utils import read_db
from utilities.sequencer import Sequencer

from time import perf_counter
from time import sleep





# Class used for loading data
class SequenceDataLoader_1h_int_6h_seq(tf.data.Dataset):
    def _generator(sequencer):
        while True: # infinite loop
            batch = sequencer.generate_batch()
            if not batch:
                break
            x , y = [], []
            for sequence in batch:
                # Image(T0), day, month, CLEARSKY_GHI(T0)
                x.append(zip(sequence[0]['image'],         # Image(T0),
                             sequence[0]['day'],           # Day(T0),
                             sequence[0]['month'],         # Month(T0)
                             sequence[0]['CLEARSKY_GHI'])) # CLEARSKY_GHI(T0)
                y.append(zip(sequence[0]['GHI'] - sequence[0]['CLEARSKY_GHI'], # GHI(T0) - CLEARSKY_GHI(T0)
                             sequence[1]['GHI'] - sequence[1]['CLEARSKY_GHI'], # GHI(T0+1) - CLEARSKY_GHI(T0+1)
                             sequence[2]['GHI'] - sequence[2]['CLEARSKY_GHI'], # GHI(T0+3) - CLEARSKY_GHI(T0+3)
                             sequence[3]['GHI'] - sequence[3]['CLEARSKY_GHI']))# GHI(T0+6) - CLEARSKY_GHI(T0+6)
                yield x, y

    def __new__(cls, sequencer):
        return tf.data.Dataset.from_generator(
            lambda: cls._generator(sequencer),
            output_types=(tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32,
                          tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32,
                          tf.dtypes.float32, tf.dtypes.float32)
            #output_shapes=((None, 5), (None, 4)),
            #args = None
        )


def benchmark(dataset, num_epochs=2):
    print('########## ENTERING benchmark...')
    total_count = 0
    print(dataset)
    start_time = perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            print('########## ITERATING Through DATASET...')
            sleep(0.5)
    tf.print("Execution time:", perf_counter() - start_time)
    tf.print("Total number of samples loaded:", total_count)

"""
Function generating sequencer object starting from dataframe
Args:
    clean_df: dataframe
    root_dir: directory for storing the data
    db_path: path for the joint table
    from_db: if 'True' gets information directly from the dataframe 
    offset: offset between two consecutive images
    seq_length: length of sequence
    batch_size: 50
Returns:
    Sequencer object
"""
def preprocess_data(clean_df,
                    root_dir='./data/preprocessed/',
                    db_path = './data/database.db',
                    from_db = False,
                    offset = 18000,
                    seq_length = 17,
                    batch_size = 50):
    db = None
    stations_names = np.unique(clean_df['station']).tolist()
    if from_db == False:
        print('Generating records...')
        records = generate_stations_dictionaries(clean_df, stations_names)

        print('Generating the joint table...')
        db = write_blocks_on_disk(records, stations_names, root_dir, db_path)
    else:
        print('Fetching information from dataframe...')
        db = pd.read_pickle('./data/database.db')

    print('Generating the mappings for the sequencer...')
    stations_mappings = read_db(db)

    print('Instantiating the sequencer...')
    return Sequencer(stations_names, stations_mappings, offset = offset, seq_length = seq_length, max_batch_size = batch_size)
