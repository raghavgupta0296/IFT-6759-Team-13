from utilities import config
from utilities.utility import load_catalog
from utilities.utility import read_ncdf
from utilities.utility import read_hdf5
from utilities.utility import map_coord_to_pixel
from utilities.utility import fetch_channel_samples
from utilities.utility import plot_and_save_image
from utilities.config import init_args
from utilities.config import CROP_SIZE
# from utilities.utility import get_datetime_attrs
from matplotlib.patches import Rectangle
from functools import lru_cache
import time
import datetime
import time, threading
import pdb
import cv2 as cv
import os, random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from utilities.utility import *

np.random.seed(5050)

avg_x = np.array([0.31950477, 283.18481332, 239.19212155, 272.73521949, 254.09056291]).reshape(1,1,5)
std_x = np.array([0.27667209, 16.24902932,  8.79865931, 20.08307892, 13.8115307]).reshape(1,1,5)

try: 
    os.mkdir("cache") 
except OSError as error: 
    print(error) 

# @lru_cache(maxsize=4096)
def load_numpy(filepath):
    path = os.path.splitext(os.path.basename(filepath))[0] + ".npz"
    path = os.path.join('npz_store',path)
    ndarray_dict = np.load(path)
    return ndarray_dict


def _generator3(dataset):
    
    if dataset.decode("utf-8")=="train":
        print("loading TRAIN data")
        with open("train_df","rb") as f:
            catalog = pickle.load(f)
#             limit = 27000
            
    elif dataset.decode("utf-8")=="valid":
        print("loading VALID data")
        with open("valid_df","rb") as f:
            catalog = pickle.load(f)
#             limit = 25000
    else:
        print("invalid train/valid string")

#     to be done while making df
#     catalog = catalog.sample(frac=1,random_state=5050)
#     i = 0
    tq = tqdm(total=len(catalog))
    for idx, row in catalog.iterrows():
#         i+=1
#         if i==limit:
#             break
        tq.update(1)
        try:
            if row.ncdf_path == "nan":
                continue
            
            # 0.0001 s 
            if not ((row.BND_DAYTIME==1) | (row.TBL_DAYTIME==1) | (row.DRA_DAYTIME==1) | (row.FPK_DAYTIME==1) | (row.GWN_DAYTIME==1) | (row.PSU_DAYTIME==1) | (row.SXF_DAYTIME==1)):
                continue

            # 0.05 s
            samples = load_numpy(row['hdf5_8bit_path'])
            offset_idx = row['hdf5_8bit_offset']
            
            timedelta_rows = [catalog[catalog.index==(idx+datetime.timedelta(hours=i))] for i in [0,1,3,6]]
            ss = ["BND","TBL","DRA","FPK","GWN","PSU","SXF"]
            np.random.shuffle(ss)
            for station_i in ss:
                if row[[station_i + "_DAYTIME"]][0]==0:
                    continue
                else:
                    #  GHI_0 = row[station_i + "_GHI"]
                    # train_df[train_df.index == train_df.index[0]+datetime.timedelta(hours=1)]
                    # pdb.set_trace()
                    GHIs = [i[station_i + "_GHI"].values[0] for i in timedelta_rows]
                    CS_GHIs = [i[station_i + "_CLEARSKY_GHI"].values[0] for i in timedelta_rows]
                    y =  np.array(GHIs) - np.array(CS_GHIs)

                    # 0.05 s
                    sample = samples[station_i]
                    x = sample[offset_idx].swapaxes(0,1).swapaxes(1,2)
                    x = (x - avg_x)/std_x
                    yield (x,y)
        except Exception as e:
#             when an offset not in training dataset, it raises error in finding future GHIs
            print(e)
            print("****** in except ******")
            continue

class SimpleDataLoader2(tf.data.Dataset):

    def __new__(cls, path):

        return tf.data.Dataset.from_generator(
            _generator3,
            args=([path]),
            output_types=(tf.float32,tf.float32),
            output_shapes=(
               tf.TensorShape([70, 70, 5]),
               tf.TensorShape([4])
               )
            ).batch(128).cache(filename="cache/keras_cache_"+path).prefetch(tf.data.experimental.AUTOTUNE)

# class SequenceDataLoader3(tf.data.Dataset):

#     def __new__(cls, args, path):

#         return tf.data.Dataset.from_generator(
#             _generator2,
#             args=([path]),
#             output_types=(
#                tf.float32,
#                tf.float32,),
#             output_shapes=(
#                tf.TensorShape([5, 70, 70, 5]),
#                tf.TensorShape([4]),
#                )).prefetch(args.batch_size).batch(args.batch_size)
#             # output_shapes=(tf.TensorShape((70, 70, 5)), tf.TensorShape((1, ))),
#             # args=(args,catalog)

if __name__ == "__main__":

    for epoch in range(1):
        print("EPOCH ", epoch)
        
        sdl_train = SimpleDataLoader2("train")
        sdl_valid = SimpleDataLoader2("valid")

        for x,y in sdl_train:
            print(x.shape,y.shape)
            continue
            
        for x,y in sdl_valid:
            print(x.shape,y.shape)
            continue
            