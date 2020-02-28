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

import datetime
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
from sklearn.utils import shuffle

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
        zero = False
        if not np.any(x):
            zero = True
        img = (x - avg_x)/std_x
        return img,y,zero

    unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())
    # print(unique_paths,type(unique_paths))
    epochs = args.epochs
    zero_img_count = 0
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
                    x,y,z = preprocess(x,y)
                    # pdb.set_trace()
                    if z:
                        zero_img_count += 1
                        if z % 10000 == 0:
                            print("Zero img count:",zero_img_count)
                        continue
                    else:
                        yield x,y
                    # yield {
                    #    # 'station_name': 'tf.string',
                    #    'images': x,
                    #    # 'csky_ghi': 0,
                    #    # 'ghi': 0,
                    #    'y': y}
        print("Zero img count:",zero_img_count)

# complex dataloader
def _generator2(path):
    args = init_args()
    catalog = load_catalog(path)

    def preprocess(x):
        zero = False
        if not np.any(x):
            zero = True
        img = (x - avg_x)/std_x
        return img,zero
    print("starting generator again...")
    unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())
    # print(unique_paths,type(unique_paths))
    epochs = args.epochs
    zero_img_count = 0
    k_sequences = 0
    GHI_sequence_steps = [4,8,12] # in the future, in addition to T0
    GHI_sequence_steps_reverse = [24,20,12,0]
    img_sequence_step = 2

    for i in range(1):
        np.random.shuffle(unique_paths)
        # print(shuffled)
        for path in unique_paths:
            # samples = fetch_all_samples_hdf5(args,path)
            try:
                samples = load_numpy(path)
            except Exception as e:
                continue
            X = [];Y = []
            grouped = catalog[path == catalog.hdf5_8bit_path]
            for station in args.station_data.keys():
                # print("I am here")
                df = grouped[grouped.station == station]
                argsort = np.argsort(df['hdf5_8bit_offset'].values)
                offsets_0 = df['hdf5_8bit_offset'].values[argsort]

                matching_offsets_imgs = offsets_0
                for i in range(k_sequences):
                    matching_offsets_imgs = np.intersect1d(matching_offsets_imgs, matching_offsets_imgs + img_sequence_step )
                # print("matching offsets",matching_offsets_imgs)
                # For GHIs
                matching_offsets_GHIs = matching_offsets_imgs
                for GHI_sequence_step in GHI_sequence_steps:
                    matching_offsets_GHIs = np.intersect1d(matching_offsets_GHIs, matching_offsets_GHIs + GHI_sequence_step)
                # print("matching offsets_GHIS",matching_offsets_GHIs)
                GHI_pairs_list = []
                CS_GHI_pairs_list = []
                y_pairs_list = []
                for i, GHI_sequence_step in enumerate(GHI_sequence_steps_reverse):
                    GHI_vals = df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs - GHI_sequence_step)].GHI.values
                    CS_GHI_vals = df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs - GHI_sequence_step)].CLEARSKY_GHI.values
                    GHI_pairs_list.append(GHI_vals)
                    CS_GHI_pairs_list.append(CS_GHI_vals)
                    y = CS_GHI_vals - GHI_vals
                    y_pairs_list.append(y)

                GHI_pairs = zip(*GHI_pairs_list)
                CS_GHI_pairs = zip(*CS_GHI_pairs_list)
                y_pairs = zip(*y_pairs_list)
                
                # iso_dt = df[df.hdf5_8bit_offset.isin(matching_offsets_imgs)]['iso-datetime'].tolist()
                # date_time_attrs = [get_datetime_attrs(dt) for dt in iso_dt] 

                offsets_pairs_list = []
                for i in range(k_sequences+1):
                    offsets_pairs_list.append(matching_offsets_imgs - (k_sequences + i)*img_sequence_step)
                offsets_pairs_list.append(matching_offsets_imgs)
                offset_pairs = zip(*offsets_pairs_list)

                GHIs_0 = df[df.hdf5_8bit_offset.isin(offsets_0)].GHI.values
                CS_GHI_0 = df[df.hdf5_8bit_offset.isin(offsets_0)].CLEARSKY_GHI.values
                y_0 = CS_GHI_0 - GHIs_0
                # example_pair = zip(offset_pairs, date_time_attrs, CS_GHI_pairs, GHI_pairs)
                # print(list(offset_pairs), list(CS_GHI_pairs), list(GHI_pairs), list(y_pairs))
                example_pair = zip(offset_pairs, CS_GHI_pairs, GHI_pairs, y_pairs)
                # if not (len(offset_pairs) == len(CS_GHI_pairs) == len(GHI_pairs) == len(y_pairs)):
                #     print("golmaal hai bhai sab golmaal hai")
                # print(list(example_pair))
                sample = samples[station]
                for offsets, CS_GHIs, GHIs, ys in example_pair:
                    imgs = []
                    for offset in offsets:
                        img = sample[offset].swapaxes(0,1).swapaxes(1,2)
                        img, status = preprocess(img)
                        imgs.append(img)
                    # img_1 = sample[offset_1].swapaxes(0,1).swapaxes(1,2)
                    # img_0 = sample[offset_0].swapaxes(0,1).swapaxes(1,2)
                    if k_sequences == 0:
                        imgs = imgs[0]
                    if False:
                        a = (imgs, date_time_pair, CS_GHIs)
                        yield (imgs, date_time_pair, CS_GHIs), (GHIs)
                    else:
                        # print("yielding")
                        X.append(imgs);Y.append(ys)
                        # yield imgs, ys
            # np.random.shuffle(X)
            # np.random.shuffle(Y)
            X, Y = shuffle(X, Y, random_state=0)
            for i,j in zip(X,Y):
                yield i,j
        print("Zero img count:",zero_img_count)

def just_call_gen():
    for i in _generator():
        a = i['images']

# simple data loader, shuffled data
def _generator3(path):
    args = init_args()
    catalog = load_catalog(path)

    def preprocess(x,y=None):
        zero = False
        if not np.any(x):
            zero = True
        img = (x - avg_x)/std_x
        return img,y,zero

    for index in tqdm(range(0,len(catalog),200)):
        rows = catalog[index:index+200]

        for idx, row in rows.iterrows():
            # print(row)
            # pdb.set_trace()
            
            if row.ncdf_path == "nan":
                continue
            samples = load_numpy(row['hdf5_8bit_path'])
            offset_idx = row['hdf5_8bit_offset']
            # continue
            timedelta_rows = [catalog[catalog.index==(idx+datetime.timedelta(hours=i))] for i in [0,1,3,6]]
            # CS_GHIs = [catalog[catalog.index==(idx+datetime.timedelta(hours=i))][station_i + "_CLEARSKY_GHI"].values[0] for i in [0,1,3,6]]
            for station_i in args.station_data.keys():
                sample = samples[station_i]
                if row[[station_i + "_GHI"]].isnull()[0]:
                    continue
                elif row[[station_i + "_DAYTIME"]][0]==0:
                    continue
                else:
                    GHI_0 = row[station_i + "_GHI"]
                    # train_df[train_df.index == train_df.index[0]+datetime.timedelta(hours=1)]
                    # pdb.set_trace()
                    GHIs = [i[station_i + "_GHI"].values[0] for i in timedelta_rows]
                    CS_GHIs = [i[station_i + "_CLEARSKY_GHI"].values[0] for i in timedelta_rows]
                    y = np.array(CS_GHIs) - np.array(GHIs)
                    if np.isnan(np.sum(y)):
                        continue
                    # ini = time.time()
                    # print(station_coords)
                    imgs = []
                    x = sample[offset_idx].swapaxes(0,1).swapaxes(1,2)
                    # print(y)
                    x = preprocess(x)[0]
                    continue
                    yield x,y

# simple data loader, shuffled data but FAST!
def _generator4(path):
    args = init_args()
    catalog = load_catalog(path)

    def preprocess(x,y=None):
        zero = False
        if not np.any(x):
            zero = True
        img = (x - avg_x)/std_x
        return img,y,zero

    unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())

    X = []; Y = []

    for path in tqdm(unique_paths):
        grouped = catalog[path == catalog.hdf5_8bit_path]
        if path == "nan":
            continue
        try:
            samples = load_numpy(path)
        except Exception as e:
            continue
        stats = ["BND","TBL","DRA","FPK","GWN","PSU","SXF"]
        # stat_array = [samples[i] for i in stats]

        for i,station_i in enumerate(args.station_data.keys()):
            # sample = samples[i]
            sample = samples[station_i]
            # try:
            #     rows = grouped[grouped.station == station_i]
            # except Exception as e:
            #     print(grouped.columns.values)
            for idx,row in grouped.iterrows():
                if row[[station_i + "_GHI"]].isnull()[0]:
                    continue
                elif not row[[station_i + "_DAYTIME"]][0]:
                    continue
                else:
                    offset_idx = row['hdf5_8bit_offset']
                    timedelta_rows = [grouped[grouped.index == (idx+datetime.timedelta(hours=i))] for i in [0,1,3,6]]
                    if timedelta_rows[-1].empty:
                        continue
                    # GHI_0 = row[station_i + "_GHI"]
                    # train_df[train_df.index == train_df.index[0]+datetime.timedelta(hours=1)]
                    # pdb.set_trace()
                    try:
                        GHIs = [i[station_i + "_GHI"].values[0] for i in timedelta_rows]
                    except Exception as e:
                        print(timedelta_rows)
                    CS_GHIs = [i[station_i + "_CLEARSKY_GHI"].values[0] for i in timedelta_rows]
                    y = np.array(CS_GHIs) - np.array(GHIs)
                    if np.isnan(np.sum(y)):
                        continue
                    # ini = time.time()
                    # print(station_coords)
                    imgs = []
                    x = sample[offset_idx].swapaxes(0,1).swapaxes(1,2)
                    # print(y)
                    x = preprocess(x)[0]
                    # print("adding")
                    continue
                    yield x,y        
        #####################################
        # for idx, row in grouped.iterrows():
        #     if row.ncdf_path == "nan":
        #         break
        #     timedelta_rows = [grouped[grouped.index==(idx+datetime.timedelta(hours=i))] for i in [0,1,3,6]]
        #     # CS_GHIs = [catalog[catalog.index==(idx+datetime.timedelta(hours=i))][station_i + "_CLEARSKY_GHI"].values[0] for i in [0,1,3,6]]
        #     offset_idx = row['hdf5_8bit_offset']

        #     for i,station_i in enumerate(args.station_data.keys()):
        #         # sample = samples[i]
        #         sample = stat_array[i]
        #         if row[[station_i + "_GHI"]].isnull()[0]:
        #             continue
        #         elif row[[station_i + "_DAYTIME"]][0]==0:
        #             continue
        #         else:
        #             GHI_0 = row[station_i + "_GHI"]
        #             # train_df[train_df.index == train_df.index[0]+datetime.timedelta(hours=1)]
        #             # pdb.set_trace()
        #             GHIs = [i[station_i + "_GHI"].values[0] for i in timedelta_rows]
        #             CS_GHIs = [i[station_i + "_CLEARSKY_GHI"].values[0] for i in timedelta_rows]
        #             y = np.array(CS_GHIs) - np.array(GHIs)
        #             if np.isnan(np.sum(y)):
        #                 continue
        #             # ini = time.time()
        #             # print(station_coords)
        #             imgs = []
        #             x = sample[offset_idx].swapaxes(0,1).swapaxes(1,2)
        #             # print(y)
        #             x = preprocess(x)[0]
        #             continue
        #             yield x,y

    # unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())
    # # print(unique_paths,type(unique_paths))
    # epochs = args.epochs
    # zero_img_count = 0
    # for i in range(1):
    #     np.random.shuffle(unique_paths)
    #     # print(shuffled)
    #     for path in unique_paths:
    #         # samples = fetch_all_samples_hdf5(args,path)
    #         try:
    #             samples = load_numpy(path)
    #         except Exception as e:
    #             continue
    #         grouped = catalog[path == catalog.hdf5_8bit_path]
    #         for station in args.station_data.keys():
    #             df = grouped[grouped.station == station]
    #             argsort = np.argsort(df['hdf5_8bit_offset'].values)
    #             offsets_0 = df['hdf5_8bit_offset'].values[argsort]

    #             GHIs_0 = df[df.hdf5_8bit_offset.isin(offsets_0)].GHI.values
    #             CS_GHI_0 = df[df.hdf5_8bit_offset.isin(offsets_0)].CLEARSKY_GHI.values
    #             y_0 = CS_GHI_0 - GHIs_0

    #             sample = samples[station]
    #             for i in range(offsets_0.shape[0]):
    #                 x = sample[i].swapaxes(0,1).swapaxes(1,2)
    #                 y = y_0[i:i+1]
    #                 # print(type(x),type(y))
    #                 x,y,z = preprocess(x,y)
    #                 # pdb.set_trace()
    #                 if z:
    #                     zero_img_count += 1
    #                     if z % 10000 == 0:
    #                         print("Zero img count:",zero_img_count)
    #                     continue
    #                 else:
    #                     yield x,y
    #                 # yield {
    #                 #    # 'station_name': 'tf.string',
    #                 #    'images': x,
    #                 #    # 'csky_ghi': 0,
    #                 #    # 'ghi': 0,
    #                 #    'y': y}
    #     print("Zero img count:",zero_img_count)

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

class SimpleDataLoader2(tf.data.Dataset):

    def __new__(cls, args, path):

        return tf.data.Dataset.from_generator(
            _generator3,
            args=([path]),
            output_types=(
               tf.float32,
               tf.float32,),
            output_shapes=(
               tf.TensorShape([70, 70, 5]),
               tf.TensorShape([4]),
               )).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        # .shuffle(10000).batch(args.batch_size).cache()
            # output_shapes=(tf.TensorShape((70, 70, 5)), tf.TensorShape((1, ))),
            # args=(args,catalog)

class SimpleDataLoader4(tf.data.Dataset):

    def __new__(cls, args, path):

        return tf.data.Dataset.from_generator(
            _generator4,
            args=([path]),
            output_types=(
               tf.float32,
               tf.float32,),
            output_shapes=(
               tf.TensorShape([70, 70, 5]),
               tf.TensorShape([4]),
               )).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

class SequenceDataLoader3(tf.data.Dataset):

    def __new__(cls, args, path):

        return tf.data.Dataset.from_generator(
            _generator2,
            args=([path]),
            output_types=(
               tf.float32,
               tf.float32,),
            output_shapes=(
               # tf.TensorShape([1, 70, 70, 5]),
               tf.TensorShape([70, 70, 5]),
               tf.TensorShape([1]),
               )).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            # output_shapes=(tf.TensorShape((70, 70, 5)), tf.TensorShape((1, ))),
            # args=(args,catalog)

def SequenceDataLoader5(args, path):
    return tf.data.Dataset.from_generator(
            _generator2,
            args=([path]),
            output_types=(
               tf.float32,
               tf.float32,),
            output_shapes=(
               # tf.TensorShape([1, 70, 70, 5]),
               tf.TensorShape([70, 70, 5]),
               tf.TensorShape([4]),
               )).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

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
