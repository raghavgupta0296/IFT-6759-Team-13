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

def _generator(path):
    args = init_args()
    catalog = load_catalog(path)

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
            try:
                samples = load_numpy(path)
            except Exception as e:
                continue
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
                    yield {
                       # 'station_name': 'tf.string',
                       'images': x,
                       # 'csky_ghi': 0,
                       # 'ghi': 0,
                       'y': y}

def just_call_gen():
    for i in _generator():
        a = i['images']


class SimpleDataLoader(tf.data.Dataset):

    def __new__(cls, args, path):

        return tf.data.Dataset.from_generator(
            _generator,
            args=([path]),
            output_types={
               # 'station_name': tf.string,
               'images': tf.float32,
               # 'csky_ghi': tf.float32,
               # 'ghi': tf.float32,
               'y': tf.float32,},
            output_shapes={
               # 'station_name': tf.TensorShape([]),
               'images': tf.TensorShape([70, 70, 5]),
               # 'csky_ghi': tf.TensorShape([1]),
               # 'ghi': tf.TensorShape([1]),
               'y': tf.TensorShape([1]),
               }).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            # output_shapes=(tf.TensorShape((70, 70, 5)), tf.TensorShape((1, ))),
            # args=(args,catalog)


if __name__ == "__main__":
    # just_call_gen()
    # exit()
    args = init_args()
    for epoch in range(args.epochs):
        print("EPOCH ", epoch)
        # self.train_loss.reset_states()
        # self.valid_loss.reset_states()

        sdl_train = SimpleDataLoader(args, args.data_catalog_path)#.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # sdl_valid = SimpleDataLoader(args, catalog_val).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        tm = tqdm(total=1000)  # R! from data loader's tqdm
        # ini = time.time()

        counter = 0
        for batch in sdl_train:
            # self.train_step(images, labels,)
            tm.update(1)
            counter += 1
            if counter > 1000:
                break
            # ini = time.time()
