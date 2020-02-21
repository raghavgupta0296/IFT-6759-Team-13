import pandas as pd
import numpy as np
import pickle as pkl

#from tqdm.notebook import tqdm
from tqdm import tqdm

from utilities import dataloader
from utilities import config
from utilities.utility import create_dummy_image
from utilities.utility import dummy_crop_image
from utilities.sequencer_utils import time_in_seconds
from utilities.sequencer_utils import convert_to_epoch
from utilities.utility import generate_file_name

# To remove later
DEBUG = 1
ROOT_DIR = '.'

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
Generates the intermediate dictionaries that will be used to produce data chunks for the model
Args:
    args: Object containing arguments
    df:  original dataframe 
    list_stations:  list of stations
Returns:
    list containing training data
"""
def generate_stations_dictionaries(df, list_stations):
    # First step: organize the dataframe with respect to file paths
    grouped = group_by(df, 'hdf5_8bit_path')
    unique_paths = np.unique(df['hdf5_8bit_path'])

    # timer
    t1 = tqdm(total=len(unique_paths))
    nb_rows = 0
    records = {}
    # Creating dictionaries
    for s in list_stations:
        records[s] = []

    # Main loop
    for path in unique_paths:
        # Grouping by paths
        cropped_df = grouped[grouped.hdf5_8bit_path == path].sort_index(axis=0)
        offsets = cropped_df['hdf5_8bit_offset'].values
        # Collecting cropped images from the compressed data
        dic = dummy_crop_image(path)

        # Iterating throw stations
        for index, row in cropped_df.iterrows():
            img = None
            station_id = row['station']
            offset = row['hdf5_8bit_offset']
            try:
                img = dic[station_id][offset]
            except KeyError:
                img = create_dummy_image()

            # Generating row and adding it to the list
            df_timestamp = row['iso-datetime']
            new_row = {'iso-datetime': convert_to_epoch(df_timestamp),
                       'station': row['station'],
                       'day': df_timestamp.day,
                       'month': df_timestamp.month,
                       'local_time': time_in_seconds(df_timestamp),
                       'image': img,
                       'CLEARSKY_GHI': row['CLEARSKY_GHI'],
                       'GHI': row['GHI']}
            records[station_id].append(new_row)
            nb_rows += 1
        t1.update(1)
    print('Generated %d rows in total' % nb_rows)
    return records



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
Generates sub-lists from the main lists
Args:
    records:  list of data
    slice_size: maximum size allowed
Returns:
    List of sub-blocks
"""
def generate_blocks_from_lists(records, slice_size):
    mini_blocks = []
    slices = get_n_slices(len(records), slice_size)
    nb_slices = len(slices)
    last = len(records) - 1
    for index in reversed(slices):
        mini_blocks.insert(0, records[index:last])
        last = index
    return mini_blocks


"""
Saves blocks in the disk
Args:
    records:  list of data
    list_stations: list of stations
    root_dir: folder
    slice_size: maximum size allowed
Returns:
    Dataframe with records pointing to the files locations
"""
def write_blocks_on_disk(records, list_stations, root_dir = ROOT_DIR + '/data/preprocessed/', slice_size = 30000):
    db_list = []
    t1 = tqdm(total=len(list_stations))
    for s in list_stations:
        # Getting records
        records_list  = records[s]
        # Slicing
        mini_blocks = generate_blocks_from_lists(records_list, slice_size)
        i = 0
        for b in mini_blocks:
            i = i + 1
            filename = generate_file_name()
            filepath = root_dir + filename + '.dat'
            new_row = {'station': s,
                       'seq':i,
                       'df_path':filepath}
            db_list.append(new_row)
            # Dumping data on disk
            pkl.dump(b, open(filepath, "wb" ) )
        t1.update(1)
    db = pd.DataFrame(db_list)
    db.to_pickle(ROOT_DIR + '/database.db')
    return db



"""
Reads data-frame information from database
Args:
    records:  list of data
Returns:
    Dictionary-representation of the dataframe
"""
def read_db(db_df):
    stations_mapping = {}
    # First step: organize the dataframe with respect to stations
    grouped = group_by(db_df, 'station')
    unique_stations = np.unique(db_df['station'])

    # Iterating throw stations
    for station in unique_stations:
        stations_mapping[station] = {}
        cropped_df = grouped[grouped.station == station]
        sequences = sorted(cropped_df['seq'].values)
        for seq in sequences:
            row = cropped_df.query(f'station == "{station}" and seq=={seq}', inplace=False)
            stations_mapping[station][seq] = row['df_path'].values[0]
    return stations_mapping
