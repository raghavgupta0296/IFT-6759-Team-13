from utilities.utility import load_catalog
from utilities.utility import read_ncdf
from utilities.utility import read_hdf5
from utilities.utility import map_coord_to_pixel
from utilities.utility import fetch_channel_samples
from utilities.utility import plot_and_save_image
from utilities.config import init_args
from utilities.config import CROP_SIZE
from matplotlib.patches import Rectangle

import cv2 as cv
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# loads dataset and iterates over dataframe rows as well as hdf5 and nc files for processing
def load_dataset(args):
    catalog = load_catalog(args)
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

class FastDataLoader(tf.data.Dataset):

    def __new__(cls, args, catalog):

        return tf.data.Dataset.from_generator(
            lambda: cls._generator(args,catalog),
            output_types=(tf.float32,tf.float32)
            # args=(args,catalog)
        )

    def _generator(args, catalog):

        STEP_SIZE =1 # args.batch_size
        # STEP_SIZE = 
        START_IDX = 0
        END_IDX = STEP_SIZE*100 #len(catalog)
        
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


# loads dataset and iterates over dataframe rows as well as hdf5 and nc files for processing
def load_dataset(args):
    catalog = load_catalog(args)
    catalog = pre_process(catalog)
    
    # print(catalog)

    # data_generator = iterate_dataset(args,catalog)
    # print(data_generator.next())

    # tf_set = tf.data.Dataset.from_generator(iterate_dataset, (tf.float32,tf.float32), args=(args,catalog))
    # print(tf_set)

    sdl = FastDataLoader(args, catalog).prefetch(tf.data.experimental.AUTOTUNE).cache()
    
    for epoch in range(args.epochs):
        # iterate over epochs
        print("Epoch: %d"%epoch)
        for i,j in sdl:
            print(i.shape,j.shape)
            # print("Incoming x and y: ", i,j)

    print("hi i reached here")

    # return SimpleDataLoader(args, catalog)

def extract_at_time(time,ctlg):
    pass

def create_data_loader():
    pass

def data_loader_main():
    args = config.init_args()
    load_dataset(args)

if __name__ == "__main__":
    data_loader_main()