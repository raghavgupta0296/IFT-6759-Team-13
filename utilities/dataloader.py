from utilities.utility import load_catalog
from utilities.utility import read_ncdf
from utilities.utility import read_hdf5
from utilities.utility import map_coord_to_pixel
from utilities.utility import fetch_channel_samples
from utilities.utility import plot_and_save_image
from utilities.config import init_args
from utilities.config import CROP_SIZE
from matplotlib.patches import Rectangle

import time, threading

import cv2 as cv
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from utilities.utility import *

# loads dataset and iterates over dataframe rows as well as hdf5 and nc files for processing
def load_dataset(args):
    catalog = load_catalog(args.data_catalog_path)
    for index, row in catalog.iterrows():
        ncdf_path = row['ncdf_path']
        hdf5_8 = row['hdf5_8bit_path']
        hdf5_16 = row['hdf5_16bit_path']
        # if .nc doesn't exist, then skip example
        if row['ncdf_path'] == "nan":
            continue
        ncdf_data = read_ncdf(ncdf_path)
        h5_data = read_hdf5(hdf5_16)

        # print(ncdf_data.dimensions)
        # print(ncdf_data.variables.keys())
        # print(ncdf_data.ncattrs)

        # extracts meta-data to map station co-ordinates to pixels
        lat_min = ncdf_data.attrs['geospatial_lat_min'][0]
        lat_max = ncdf_data.attrs['geospatial_lat_max'][0]
        lon_min = ncdf_data.attrs['geospatial_lon_min'][0]
        lon_max = ncdf_data.attrs['geospatial_lon_max'][0]
        lat_res = ncdf_data.attrs['geospatial_lat_resolution'][0]
        lon_res = ncdf_data.attrs['geospatial_lon_resolution'][0]

        station_coords = []
        for sta, (lat,lon,elev) in args.station_data.items():
            # y = row data (longitude: changes across rows i.e. vertically)
            # x = column data (latitude: changes across columns i.e horizontally)
            x,y = [map_coord_to_pixel(lat,lat_min,lat_res), map_coord_to_pixel(lon,lon_min,lon_res)]
            station_coords.append([y,x])
        station_coords = np.array(station_coords)

        # reads h5 and ncdf samples
        h5_16bit_samples = fetch_channel_samples(args,h5_data,row['hdf5_16bit_offset'])
        ncdf_samples = [ncdf_data.variables[ch][0] for ch in args.channels]

        plot_and_save_image(args,station_coords,h5_16bit_samples,prefix="h5_16")
        plot_and_save_image(args,station_coords,ncdf_samples,prefix="ncdf")
        break


# pre process dataset to remove common nans in dataframe
def pre_process(dataset):
    # no night time data
    pp_dataset = dataset[(dataset.BND_DAYTIME==1) | (dataset.TBL_DAYTIME==1) | (dataset.DRA_DAYTIME==1) | (dataset.FPK_DAYTIME==1) | (dataset.GWN_DAYTIME==1) | (dataset.PSU_DAYTIME==1) | (dataset.SXF_DAYTIME==1)]
    
    # no empty path images
    pp_dataset = pp_dataset[pp_dataset.ncdf_path!="nan"]
    
    # make iso_datetime a column instead of index
    pp_dataset = pp_dataset.reset_index()
    
    # shuffle all rows of dataset 
    # !!! REMOVE FOR CONSIDERING TIME SEQUENCING ###
    # pp_dataset = pp_dataset.sample(frac=1).reset_index(drop=True)
    # pp_dataset = pp_dataset.reset_index(drop=True)

    return pp_dataset

def station_from_row(args, rows):
    x = []; y = []
    # R! vectorize this by using iloc instead of iterrows?  
    for _, row in rows.iterrows():
        ncdf_path = row['ncdf_path']
        hdf5_8 = row['hdf5_8bit_path']
        hdf5_16 = row['hdf5_16bit_path']
        # if .nc doesn't exist, then skip example
        if row['ncdf_path'] == "nan":
            continue

        if args.image_data == 'hdf5v7_8bit':
            data_handle = read_hdf5(hdf5_8)
            idx = row['hdf5_8bit_offset']
            samples = fetch_channel_samples(args,data_handle,idx)
        elif args.image_data == 'hdf5v5_16bit':
            data_handle = read_hdf5(hdf5_16)
            idx = row['hdf5_16bit_offset']
            samples = fetch_channel_samples(args,data_handle,idx)
        elif args.image_data == 'netcdf':
            data_handle = read_ncdf(ncdf_path)
            samples = [data_handle.variables[ch][0] for ch in args.channels]
        
        # print(ncdf_data.dimensions)
        # print(ncdf_data.variables.keys())
        # print(ncdf_data.ncattrs)

        # print(data_handle.keys())
        # print(data_handle['lon_LUT'])
        # print(data_handle['lon'][0])
        # print(data_handle['lat'][0])
        # print(data_handle['lat_LUT'])

        # extracts meta-data to map station co-ordinates to pixels
        station_coords = {}
        if args.image_data == 'hdf5v7_8bit' or args.image_data == 'hdf5v5_16bit':
            lats, lons = utils.fetch_hdf5_sample("lat", data_handle, idx), utils.fetch_hdf5_sample("lon", data_handle, idx)
            for sta, (lat,lon,elev) in args.station_data.items():
                # y = row data (longitude: changes across rows i.e. vertically)
                # x = column data (latitude: changes across columns i.e horizontally)
                x_coord,y_coord = [np.argmin(np.abs(lats-lat)),np.argmin(np.abs(lons-lon))]
                station_coords[sta] = [y_coord,x_coord]
                # print(x_coord,y_coord)
        else:
            lat_min = data_handle.attrs['geospatial_lat_min'][0]
            lat_max = data_handle.attrs['geospatial_lat_max'][0]
            lon_min = data_handle.attrs['geospatial_lon_min'][0]
            lon_max = data_handle.attrs['geospatial_lon_max'][0]
            lat_res = data_handle.attrs['geospatial_lat_resolution'][0]
            lon_res = data_handle.attrs['geospatial_lon_resolution'][0]

            for sta, (lat,lon,elev) in args.station_data.items():
                # y = row data (longitude: changes across rows i.e. vertically)
                # x = column data (latitude: changes across columns i.e horizontally)
                x_coord,y_coord = [map_coord_to_pixel(lat,lat_min,lat_res),map_coord_to_pixel(lon,lon_min,lon_res)]
                station_coords[sta] = [y_coord,x_coord] 
        
        # reads h5 and ncdf samples

        # h5_16bit_samples = fetch_channel_samples(args,h5_data,row['hdf5_16bit_offset'])
        # h5_16bit_samples = fetch_channel_samples(args,data_handle,row['hdf5_8bit_offset'])

        # ncdf_samples = [data_handle.variables[ch][0] for ch in args.channels]
        samples = np.array(samples)
        print("sample shape:",samples.shape)
        # R! question: -ve large values in ncdf_samples?
        # print(ncdf_samples)

        # h5_16bit_samples = np.array(h5_16bit_samples)
        # print(h5_16bit_samples)
        # print(type(h5_16bit_samples))
        # print(h5_16bit_samples.shape)

        # plot_and_save_image(args,station_coords,h5_16bit_samples,prefix="h5_16")
        # plot_and_save_image(args,station_coords,ncdf_samples,prefix="ncdf")

        for station_i in config.STATION_NAMES:
        # station_i = 'FPK'
            if row[[station_i+"_GHI"]].isnull()[0]:
                # print("[INFO] GHI is null for station ", station_i)
                continue
            elif row[[station_i+"_DAYTIME"]][0]==0:
                # print("[INFO] Night for station ", station_i)
                continue
            # print(station_i)

            y.append(row[station_i+"_GHI"])
            # ini = time.time()
            # print(station_coords)
            x.append(crop_station_image(station_i,samples,station_coords))
            # print("cropping time: ", time.time()-ini)
    return x,y

# crop station image from satellite image of size CROP_SIZE
def crop_station_image(station_i,sat_image,station_coords):

    # R! check  crop correct positions? and also if lower origin needs to be taken before manual cropping
    
    crop_size = args.CROP_SIZE

    # fig,ax = plt.subplots(1)
    # ax.imshow(sat_image[0], cmap='bone')
    # rect = Rectangle((station_coords[station_i][0]-(crop_size//2),station_coords[station_i][1]-(crop_size//2)),crop_size,crop_size,linewidth=1,fill=True,edgecolor='r',facecolor='none')
    # ax.add_patch(rect)
    # plt.scatter(station_coords[station_i][0],station_coords[station_i][1])
    # plt.savefig("check_crop.png")
    
    # print("in crop station image: ", station_coords[station_i][1]-(crop_size//2)," - " , (station_coords[station_i][1]+(crop_size//2)))
    margin = crop_size//2
    lat_mid = station_coords[station_i][1]
    lon_mid = station_coords[station_i][0]
    crop = sat_image[
        :, 
        lat_mid-margin:lat_mid+margin, 
        lon_mid-margin:lon_mid+margin, 
        ]

    if crop.shape!=(5,crop_size,crop_size):
        print("[WARNING] crop channels shape:", station_i, [crop[i].shape for i in range(len(crop))])
    
    # plt.imshow(crop[0], cmap='bone')
    # plt.savefig("check_cropped.png")

    return crop

class SimpleDataLoader(tf.data.Dataset):

    def __new__(cls, args, catalog):

        return tf.data.Dataset.from_generator(
            lambda: cls._generator(args,catalog),
            output_types=(tf.float32,tf.float32)
            # args=(args,catalog)
        )

    def _generator(args, catalog):

        STEP_SIZE = args.batch_size
        # STEP_SIZE = 
        START_IDX = 0
        END_IDX = STEP_SIZE*3 #len(catalog)
        
        if args.debug:
            STEP_SIZE = 1
            END_IDX = STEP_SIZE*3

        for index in tqdm(range(START_IDX,END_IDX,STEP_SIZE)): 
        # while(index < len(catalog)):

            rows = catalog[ index : index+STEP_SIZE ]
            # print(rows)

            if args.debug:
                profiler = LineProfiler()
                profiled_func = profiler(station_from_row)
                try:
                    profiled_func(args, rows, x, y)
                finally:
                    profiler.print_stats()
                    profiler.dump_stats('data_loader_dump.txt')
            else:
                x,y = station_from_row(args, rows)

            x = np.array(x)
            y = np.array(y)
            print("Yielding x (shape) and y (shape) of index: ", index, x.shape,y.shape)

            yield (x,y)

def store_numpy(ndarray_dict,filepath):
    os.makedirs('npz_store',exist_ok=True)
    path = os.path.splitext(os.path.basename(filepath))[0] + ".npz"
    path = os.path.join('npz_store',path)
    np.savez(path, **ndarray_dict)

def store_pickle(ndarray_dict,filepath):
    os.makedirs('pickle_store',exist_ok=True)
    path = os.path.splitext(os.path.basename(filepath))[0] + ".dat"
    path = os.path.join('pickle_store',path)
    # np.savez(path, **ndarray_dict)
    with open(path, 'wb') as outfile:
        pickle.dump(ndarray_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath):
    path = os.path.splitext(os.path.basename(filepath))[0] + ".dat"
    path = os.path.join('pickle_store',path)
    # np.savez(path, **ndarray_dict)
    tic = time.time()
    with open(path, 'rb') as infile:
        ndarray_dict = pickle.load(infile)
    toc = time.time()
    print("pkl:",toc-tic)
    return ndarray_dict

def load_numpy(filepath):
    path = os.path.splitext(os.path.basename(filepath))[0] + ".npz"
    path = os.path.join('npz_store',path)
    tic = time.time()
    ndarray_dict = np.load(path)
    toc = time.time()
    print("npz:",toc-tic)
    return ndarray_dict

# generates t0-1 and t0 data
class SequenceDataLoader(tf.data.Dataset):

    def __new__(cls, args, catalog):

        return tf.data.Dataset.from_generator(
            lambda: cls._generator(args,catalog),
            output_types=(tf.float32,tf.float32)
            # args=(args,catalog)
        )

    def _generator(args, catalog):

        unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())
        print(unique_paths)
        STEP_SIZE =1 # args.batch_size
        # STEP_SIZE = 
        START_IDX = 0
        END_IDX = STEP_SIZE*100 #len(catalog)
        
        if args.debug:
            STEP_SIZE = 1
            END_IDX = STEP_SIZE*3

        for path in tqdm(unique_paths):

            samples = fetch_all_samples_hdf5(args,path)
            # store_numpy(samples,path)

            # continue

            # grouping by paths
            grouped = catalog[path == catalog.hdf5_8bit_path]
            offsets_1 = {} # T0 - 1
            offsets_0 = {} # T0
            example_pairs = {}
            for station in args.station_data.keys():
                df = grouped[grouped.station == station]
                argsort = np.argsort(df['hdf5_8bit_offset'].values)
                offsets_1[station] = df['hdf5_8bit_offset'].values[argsort]
                offsets_0[station] = offsets_1[station] + 4
                
                # if offsets+4 offset exists, we create pairs using those offsets+4 since T0-1 exists by definition
                matching_offsets = np.intersect1d(offsets_1[station],offsets_0[station])
                # pairs = zip(matching_offsets-4,matching_offsets)
                GHI_pairs = zip(df[df.hdf5_8bit_offset.isin(matching_offsets-4)].GHI.values, df[df.hdf5_8bit_offset.isin(matching_offsets)].GHI.values)
                offset_pairs = zip(matching_offsets-4,matching_offsets)
                
                example_pairs[station] = zip(offset_pairs,GHI_pairs)
                # print(example_pairs)
            for station,ex_pair in example_pairs.items():
                # for offset,GHI in ex_pair:
                for (offset_1,offset_0),(GHI_1,GHI_0) in ex_pair:
                # for a in ex_pair:
                    # print(list(offset))
                    img_1 = samples[station][offset_1]
                    img_0 = samples[station][offset_0]
                    yield (img_1,img_0),(GHI_1,GHI_0) 

# generates t0-1 and t0 data
class SequenceDataLoaderMemChunks(tf.data.Dataset):

    def __new__(cls, args, catalog):

        return tf.data.Dataset.from_generator(
            lambda: cls._generator(args,catalog),
            output_types=(tf.float32,tf.float32)
            # args=(args,catalog)
        )

    def _generator(args, catalog):

        unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())
        # print(unique_paths)
        STEP_SIZE =1 # args.batch_size
        # STEP_SIZE = 
        START_IDX = 0
        END_IDX = STEP_SIZE*100 #len(catalog)
        
        if args.debug:
            STEP_SIZE = 1
            END_IDX = STEP_SIZE*3
        counter = 0

        k_sequences = args.k_sequences # in the past, addition to T0 
        img_sequence_step = args.img_sequence_step
        GHI_sequence_steps = [4,12,24] # in the future, in addition to T0
        GHI_sequence_steps = GHI_sequence_steps[:args.future_ghis]
        GHI_sequence_steps.reverse()
        for path in tqdm(unique_paths):

            # samples = fetch_all_samples_hdf5(args,path)
            # store_numpy(samples,path)
            samples = load_numpy(path)
            # continue
            # grouping by paths
            grouped = catalog[path == catalog.hdf5_8bit_path]
            offsets_1 = {} # T0 - 1
            offsets_0 = {} # T0
            example_pairs = {}
            offsets_list = []
            for station in args.station_data.keys():
                df = grouped[grouped.station == station]
                argsort = np.argsort(df['hdf5_8bit_offset'].values)
                offsets_1[station] = df['hdf5_8bit_offset'].values[argsort]
                offsets_0[station] = offsets_1[station] + img_sequence_step
                matching_offsets_imgs = df['hdf5_8bit_offset'].values[argsort]
                for i in range(k_sequences):
                    matching_offsets_imgs = np.intersect1d(matching_offsets_imgs, matching_offsets_imgs + img_sequence_step )
                # offsets_plus_1[station] = offsets_1[station] + 8
                
                # if offsets+4 offset exists, we create pairs using those offsets+4 since T0-1 exists by definition
                # matching_offsets_imgs = np.intersect1d(offsets_1[station],offsets_0[station])
                # pairs = zip(matching_offsets-4,matching_offsets)

                # For GHIs
                
                matching_offsets_GHIs = matching_offsets_imgs
                for GHI_sequence_step in GHI_sequence_steps:
                    matching_offsets_GHIs = np.intersect1d(matching_offsets_GHIs, matching_offsets_GHIs + GHI_sequence_step)
                # matching_offsets_GHIs = np.intersect1d(matching_offsets_imgs, matching_offsets_imgs + GHI_sequence_step)

                GHI_pairs_list = []
                # CS_GHI_pairs_list = []
                for GHI_sequence_step in GHI_sequence_steps:
                    GHI_pairs_list.append(df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs - GHI_sequence_step)].GHI.values)
                GHI_pairs_list.append(df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs)].GHI.values)
                #     CS_GHI_pairs_list.append(df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs - GHI_sequence_step)].CLEARSKY_GHI.values)
                # GHI_pairs_list.append(df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs)].CLEARSKY_GHI.values)

                # GHI_pairs = zip(df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs - GHI_sequence_step )].GHI.values, 
                #     df[df.hdf5_8bit_offset.isin(matching_offsets_GHIs)].GHI.values)
                GHI_pairs = zip(*GHI_pairs_list)

                # for images
                # offset_pairs = zip(matching_offsets-4,matching_offsets)
                offsets_pairs_list = []
                for i in range(k_sequences):
                    offsets_pairs_list.append(matching_offsets_imgs - (k_sequences + i))
                offsets_pairs_list.append(matching_offsets_imgs)
                offset_pairs = zip(*offsets_pairs_list)
                # offset_pairs = zip(matching_offsets_imgs - img_sequence_step, matching_offsets_imgs)

                # example_pairs[station] = zip(offset_pairs,GHI_pairs)
                sample = samples[station]
                example_pair = zip(offset_pairs, GHI_pairs)
                # for (offset_1,offset_0),(GHI_0,GHI_plus_1) in example_pair:
                for offsets,GHIs in example_pair:
                    # for a in ex_pair:
                    # print(list(offset))
                    imgs = []
                    for offset in offsets:
                        img = sample[offset].swapaxes(0,1).swapaxes(1,2)
                        imgs.append(img)
                    # img_1 = sample[offset_1].swapaxes(0,1).swapaxes(1,2)
                    # img_0 = sample[offset_0].swapaxes(0,1).swapaxes(1,2)
                    # tic = time.time()
                    # print(tic-toc)
                    # print(counter)
                    # counter += 1
                    yield imgs,GHIs
                    # yield (img_1,img_0),(GHI_0)
                # print(example_pairs)

def iterate_and_fetch_all_samples_hdf5(args,paths):
    for path in paths:
        samples = fetch_all_samples_hdf5(args,path)
        store_numpy(samples,path)
        store_pickle(samples,path)
        load_numpy(samples,path)
        load_pickle(samples,path)
        print("stored %s"%path)

# generates t0-1 and t0 data
class SequenceDataLoaderThreaded(tf.data.Dataset):

    def __new__(cls, args, catalog):

        return tf.data.Dataset.from_generator(
            lambda: cls._generator(args,catalog),
            output_types=(tf.float32,tf.float32)
            # args=(args,catalog)
        )

    def _generator(args, catalog):

        unique_paths = pd.unique(catalog['hdf5_8bit_path'].values.ravel())
        print(unique_paths)
        STEP_SIZE =1 # args.batch_size
        # STEP_SIZE = 
        START_IDX = 0
        END_IDX = STEP_SIZE*100 #len(catalog)
        
        if args.debug:
            STEP_SIZE = 1
            END_IDX = STEP_SIZE*3
        
        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()
        num_threads = 5
        threads = []      
  
        unique_paths = unique_paths.tolist()
        size = len(unique_paths)

        for thread_index in range(num_threads):
            from_ = size // num_threads * (thread_index)
            upto = size // num_threads * (thread_index + 1)
            args1 = (args,unique_paths[from_:upto])
            t = threading.Thread(target=iterate_and_fetch_all_samples_hdf5, args=args1)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print("exiting")
        return
        # samples = fetch_all_samples_hdf5(args,path)
        # store_numpy(samples,path)
        #     continue
        # grouping by paths
        grouped = catalog[path == catalog.hdf5_8bit_path]
        offsets_1 = {} # T0 - 1
        offsets_0 = {} # T0
        example_pairs = {}
        for station in args.station_data.keys():
            df = grouped[grouped.station == station]
            argsort = np.argsort(df['hdf5_8bit_offset'].values)
            offsets_1[station] = df['hdf5_8bit_offset'].values[argsort]
            offsets_0[station] = offsets_1[station] + 4
            
            # if offsets+4 offset exists, we create pairs using those offsets+4 since T0-1 exists by definition
            matching_offsets = np.intersect1d(offsets_1[station],offsets_0[station])
            # pairs = zip(matching_offsets-4,matching_offsets)
            GHI_pairs = zip(df[df.hdf5_8bit_offset.isin(matching_offsets-4)].GHI.values, df[df.hdf5_8bit_offset.isin(matching_offsets)].GHI.values)
            offset_pairs = zip(matching_offsets-4,matching_offsets)
            
            example_pairs[station] = zip(offset_pairs,GHI_pairs)
            # print(example_pairs)
        for station,ex_pair in example_pairs.items():
            # for offset,GHI in ex_pair:
            for (offset_1,offset_0),(GHI_1,GHI_0) in ex_pair:
            # for a in ex_pair:
                # print(list(offset))
                img_1 = samples[station][offset_1]
                img_0 = samples[station][offset_0]
                yield (img_1,img_0),(GHI_1,GHI_0) 

# loads dataset and iterates over dataframe rows as well as hdf5 and nc files for processing
def load_dataset(args):
    catalog = load_catalog(args.data_catalog_path)
    catalog = pre_process(catalog)
    
    # print(catalog)

    # data_generator = iterate_dataset(args,catalog)
    # print(data_generator.next())

    # tf_set = tf.data.Dataset.from_generator(iterate_dataset, (tf.float32,tf.float32), args=(args,catalog))
    # print(tf_set)

    sdl = SimpleDataLoader(args, catalog).prefetch(tf.data.experimental.AUTOTUNE).cache()
    
    for epoch in range(args.epochs):
        # iterate over epochs
        print("Epoch: %d"%epoch)
        for i,j in sdl:
            print(i.shape,j.shape)
            # print("Incoming x and y: ", i,j)

    print("hi i reached here")

# loads dataset and iterates over dataframe rows as well as hdf5 and nc files for processing
def load_dataset_seq(args):
    catalog = load_catalog(args.data_catalog_path)
    # catalog = pre_process(catalog)
    
    # print(catalog)

    # data_generator = iterate_dataset(args,catalog)
    # print(data_generator.next())

    # tf_set = tf.data.Dataset.from_generator(iterate_dataset, (tf.float32,tf.float32), args=(args,catalog))
    # print(tf_set)

    sdl = SequenceDataLoaderThreaded(args, catalog)
    # sdl = SequenceDataLoaderMemChunks(args, catalog)
    # sdl = sdl.map(lambda x,y: (x,y), num_parallel_calls=4)
    sdl = sdl.prefetch(tf.data.experimental.AUTOTUNE)
    
    for epoch in range(args.epochs):
        # iterate over epochs
        print("Epoch: %d"%epoch)
        for (img_1,img_0),(GHI_1,GHI_0) in sdl:
            # print(img_1.shape,img_0.shape)
            pass

    print("hi i reached here")

def extract_at_time(time,ctlg):
    pass

def create_data_loader():
    pass

def data_loader_main():
    args = init_args()
    load_dataset_seq(args)

if __name__ == "__main__":
    data_loader_main()