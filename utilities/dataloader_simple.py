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

import time, threading
import pdb
import cv2 as cv
import os, random
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from utilities.utility import *

avg_x = np.array([0.31950477, 283.18481332, 239.19212155, 272.73521949, 254.09056291]).reshape(1,1,5)
std_x = np.array([0.27667209, 16.24902932,  8.79865931, 20.08307892, 13.8115307]).reshape(1,1,5)

def load_numpy(filepath):
    path = os.path.splitext(os.path.basename(filepath))[0] + ".npz"
    path = os.path.join('npz_store',path)
    ndarray_dict = np.load(path)
    return ndarray_dict

class SimpleDataLoader(tf.data.Dataset):

    def __new__(cls, args, catalog):

        return tf.data.Dataset.from_generator(
            lambda: cls._generator(args,catalog),
            output_types=(tf.float32,tf.float32),
            output_shapes=(tf.TensorShape((70, 70, 5)), tf.TensorShape((1, ))),
            # args=(args,catalog)
        )
    def _generator(args, catalog):

        def preprocess(x,y):
            if not np.any(x):
                print("zero img in training")
            img = (x - avg_x)/std_x
            return img,y

        unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())
        # print(unique_paths,type(unique_paths))
        epochs = args.epochs
        for i in range(1):
            np.random.shuffle(unique_paths)
            # print(shuffled)
            for path in unique_paths:
                # samples = fetch_all_samples_hdf5(args,path)
                samples = load_numpy(path)

                grouped = catalog[path == catalog.hdf5_8bit_path]
                for station in args.station_data.keys():
                    df = grouped[grouped.station == station]
                    argsort = np.argsort(df['hdf5_8bit_offset'].values)
                    offsets_0 = df['hdf5_8bit_offset'].values[argsort]

                    GHIs_0 = df[df.hdf5_8bit_offset.isin(offsets_0)].GHI.values
                    CS_GHI_0 = df[df.hdf5_8bit_offset.isin(offsets_0)].CLEARSKY_GHI.values
                    y_0 = CS_GHI_0 - GHIs_0

                    sample = samples[station]
                    for i in range(offsets_0.shape[0]):
                        x = sample[i].swapaxes(0,1).swapaxes(1,2)
                        y = y_0[i:i+1]
                        # print(type(x),type(y))
                        # x,y = preprocess(x,y)
                        # pdb.set_trace()
                        yield x,y
