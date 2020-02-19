import pandas as pd
import numpy as np

#from tqdm.notebook import tqdm
from tqdm import tqdm

from utilities import dataloader
from utilities import config
from utilities.utility import create_dummy_image
from utilities.utility import dummy_crop_image
from utilities.sequencer_utils import time_in_seconds

"""
Groups the dataframe per column and sorts it with respect to timestamps
Args:
    start_time:  timestamp 
Returns:
    Dataframe chunk containing rows with specified path
"""
def group_by(df, column):
    try:
        grouped = df.groupby(column, as_index=False, sort=False).apply(lambda x: x.sort_index(ascending=True))
    except KeyError:
        return None
    return grouped


"""
Generates the intermediate dataframe that will be used to produce data chunks for the model
Args:
    df:  original dataframe 
Returns:
    Dataframe containing training data
"""
def generate_intermediate_dataframe(df):
    # First step: organize the dataframe with respect to file paths
    grouped = group_by(df, 'hdf5_8bit_path')
    unique_paths = np.unique(df['hdf5_8bit_path'])

    # timer
    t1 = tqdm(total=len(unique_paths))
    nb_rows = 0
    list_of_records = []
    for path in unique_paths:
        # print('### processing path \'%s\'' %path)
        # Grouping by paths
        cropped_df = grouped[grouped.hdf5_8bit_path == path].sort_index(axis=0)
        offsets = cropped_df['hdf5_8bit_offset'].values
        # Collecting cropped images from the compressed data

        # Bhavya's note: please pass the args instance in the following function
        # or if instantiated earlier, can pass it as an argument from the calling fn
        args = config.init_args() # <- instantiated like this if not done
        dic = dataloader.fetch_all_samples_hdf5(args,path)
        # dic = create_dummy_image()

        # Iterating throw stations
        for index, row in cropped_df.iterrows():
            img = None
            station_id = row['station']
            offset = row['hdf5_8bit_offset']
            try:
                img = dic[station_id][offset]
            except KeyError:
                img = create_dummy_image()

            # Creating the row and adding it to the dataframe
            df_timestamp = row['iso-datetime']
            new_row = {'iso-datetime': df_timestamp,
                       'day': df_timestamp.day,
                       'month': df_timestamp.month,
                       'local_time': time_in_seconds(df_timestamp),
                       'station': station_id,
                       'image': img,
                       'CLEARSKY_GHI': row['CLEARSKY_GHI'],
                       'GHI': row['GHI']}
            list_of_records.append(new_row)
            nb_rows += 1
        t1.update(1)
    print('Generated %d rows in total' % nb_rows)
    return pd.DataFrame(list_of_records)



"""
Gets list of indexes given a size and a slice offset
Args:
    total_size:  size to split
    slice: size of each chunk
Returns:
    list of indexes corresponding to the splits
"""
def get_n_slices(total_size, slice):
    indexes = []
    nb_slices = int(total_size / slice)
    for idx in range(nb_slices + 1):
        indexes.append(idx * slice)
    return indexes


"""
Splits dataframe to smaller dataframes
Args:
    dataframe:  main dataframe
    slice_length: size of each subset
Returns:
    list of blocks containing smaller subsets
    
TODO: function needs to be debugged. pandas dataframe slicing does not work well
"""
def generate_blocks_from_df(dataframe, slice_length):
    mini_blocks = []
    slices = get_n_slices(dataframe.size, slice_length)
    nb_slices = len(slices)
    print('slices are :', slices)
    last = dataframe.size
    for index in reversed(slices):
        print('first:', index, ', last:', last)
        print('Getting slice [%d, %d]' %(index, last))
        block = dataframe[index:last]
        print('Got a block of size %d' %block.size)
        mini_blocks.insert(0, block)
        last = index
    return mini_blocks


"""
Saves blocks in the disk
Args:
    blocks_dic:  list of dataframes
    root_dir: directory where the blocks are to be saved
Returns:
    Dataframe with records pointing to the files locations
"""
def write_blocks_on_disk(blocks_dic, root_dir='./'):
    block_list = []
    for station, blocks in block_dic.items():
        for i, b in enumerate(blocks):
            filename = root_dir + get_file_name() + '.pkl'
            new_row = {'station': station,
                   'seq': i + 1,
                   'path':filename}
            # Dumping data on disk
            b.to_pickle(filename)
    # Creating the dataframe
    db = pd.DataFrame(block_list)
    return db