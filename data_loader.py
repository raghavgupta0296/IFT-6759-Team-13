import pickle
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py, h5netcdf
from tqdm import tqdm
import pdb
import time

import utils
import config

# loads the pickle dataframe containing data paths and targets information
def load_catalog(args):
    f = open(args.data_catalog_path,"rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

# loads an hdf5 file
def read_hdf5(hdf5_path):
    h5_data = h5py.File(hdf5_path, "r")
    return h5_data

# loads a netcdf file
def read_ncdf(ncdf_path):
    ncdf_data = h5netcdf.File(ncdf_path, 'r')
    return ncdf_data

# maps a physical co-ordinate to image pixel
def map_coord_to_pixel(coord,min_coord,res):
    x = int(abs(min_coord - coord)//res)
    return x

# extract images of all the 5 channels given offset and data handle
def fetch_channel_samples(args,h5_data_handle,hdf5_offset):
    channels = args.channels
    sample = [utils.fetch_hdf5_sample(ch, h5_data_handle, hdf5_offset) for ch in channels]
    return sample

# saves images of 5 channels with plotted mapped co-ordinates
def plot_and_save_image(args,station_coords,samples,prefix="0"):
    all_coords = np.array(list(station_coords.values()))
    cmap='bone'
    for sample,ch in zip(samples,args.channels):
        plt.imshow(sample,origin='lower',cmap=cmap)
        plt.scatter(all_coords[:,0],all_coords[:,1])
        plt.savefig("sample_outputs/%s_%s.png"%(prefix,ch))

# pre process dataset to remove common nans in dataframe 
def pre_process(dataset):
    # no night time data
    pp_dataset = dataset[(dataset.BND_DAYTIME==1) | (dataset.TBL_DAYTIME==1) | (dataset.DRA_DAYTIME==1) | (dataset.FPK_DAYTIME==1) | (dataset.GWN_DAYTIME==1) | (dataset.PSU_DAYTIME==1) | (dataset.SXF_DAYTIME==1)]
    # no empty path images
    pp_dataset = pp_dataset[pp_dataset.ncdf_path!="nan"]
    # make iso_datetime a column instead of index
    pp_dataset = pp_dataset.reset_index()
    
    # shuffle all rows of dataset - REMOVE FOR CONSIDERING TIME SEQUENCING
    pp_dataset = pp_dataset.sample(frac=1).reset_index(drop=True)
    
    return pp_dataset

def station_from_row(args, rows, x, y):
    # R! vectorize this by using iloc instead of iterrows?  
    for _, row in rows.iterrows():
        ncdf_path = row['ncdf_path']
        hdf5_8 = row['hdf5_8bit_path']
        hdf5_16 = row['hdf5_16bit_path']
        # if .nc doesn't exist, then skip example
        if row['ncdf_path'] == "nan":
            continue
            
        ini = time.time()
        h5_data = read_hdf5(hdf5_16)
        print("time taken for reading hfd5 16 bit file: ", time.time()-ini)

        ini = time.time()
        h5_data = read_hdf5(hdf5_8)
        print("time taken for reading hfd5 8 bit file: ", time.time()-ini)
        
        ini = time.time()
        ncdf_data = read_ncdf(ncdf_path)
        print("time for reading ncdf file", time.time()-ini)
        
        # print(ncdf_data.dimensions)
        # print(ncdf_data.variables.keys())
        # print(ncdf_data.ncattrs)
        
        # extracts meta-data to map station co-ordinates to pixels
        ini = time.time()
        lat_min = ncdf_data.attrs['geospatial_lat_min'][0]
        lat_max = ncdf_data.attrs['geospatial_lat_max'][0]
        lon_min = ncdf_data.attrs['geospatial_lon_min'][0]
        lon_max = ncdf_data.attrs['geospatial_lon_max'][0]
        lat_res = ncdf_data.attrs['geospatial_lat_resolution'][0]
        lon_res = ncdf_data.attrs['geospatial_lon_resolution'][0]
        print("time for extracting attributes from ncdf file", time.time()-ini)

        ini = time.time()
        station_coords = {}
        # R! reading from dictionary can lead to random order of key pairs, 
        # storing datatype changed from array to dictionary
        for sta, (lat,lon,elev) in args.station_data.items():
            # y = row data (longitude: changes across rows i.e. vertically)
            # x = column data (latitude: changes across columns i.e horizontally)
            x_coord,y_coord = [map_coord_to_pixel(lat,lat_min,lat_res),map_coord_to_pixel(lon,lon_min,lon_res)]
            station_coords[sta] = [y_coord,x_coord] 
        print("time to find coordinates: ", time.time()-ini)

        # reads h5 and ncdf samples
        ini = time.time()
        h5_16bit_samples = fetch_channel_samples(args,h5_data,row['hdf5_16bit_offset'])
        print("time taken for reading hdf5 16 bit sat image: ", time.time()-ini)
        
        ini = time.time()
        h5_16bit_samples = fetch_channel_samples(args,h5_data,row['hdf5_8bit_offset'])
        print("time taken for reading hdf5 8 bit sat image: ", time.time()-ini)

        # ncdf_samples = [ncdf_data.variables[ch][0] for ch in args.channels]

        # R! question: -ve large values in ncdf_samples?
        # print(ncdf_samples)

        h5_16bit_samples = np.array(h5_16bit_samples)
        # print(h5_16bit_samples)
        # print(type(h5_16bit_samples))
        # print(h5_16bit_samples.shape)

        for station_i in config.STATION_NAMES:
        # station_i = 'FPK'
            if row[[station_i+"_GHI"]].isnull()[0]:
                print("[INFO] GHI is null for station ", station_i)
                continue
            elif row[[station_i+"_DAYTIME"]][0]==0:
                print("[INFO] Night for station ", station_i)
                continue
            # print(station_i)
            y.append(row[station_i+"_GHI"])
            ini = time.time()
            x.append(crop_station_image(station_i,h5_16bit_samples,station_coords))
            print("cropping time: ", time.time()-ini)
    return x, y

prev_station_coords = {'BND': [915, 401], 'TBL': [494, 403], 'DRA': [224, 315], 'FPK': [497, 607], 'GWN': [878, 256], 'PSU': [1176, 418], 'SXF': [709, 493]}

# crop station image from satellite image of size CROP_SIZE
def crop_station_image(station_i,sat_image,station_coords):

    # R! check  crop correct positions? and also if lower origin needs to be taken before manual cropping
    global prev_station_coords
    
    assert prev_station_coords==station_coords
    prev_station_coords = station_coords
#     print(station_coords)
    
    crop_size = config.CROP_SIZE

    # fig,ax = plt.subplots(1)
    # ax.imshow(sat_image[0], cmap='bone')
    # rect = Rectangle((station_coords[station_i][0]-(crop_size//2),station_coords[station_i][1]-(crop_size//2)),crop_size,crop_size,linewidth=1,fill=True,edgecolor='r',facecolor='none')
    # ax.add_patch(rect)
    # plt.scatter(station_coords[station_i][0],station_coords[station_i][1])
    # plt.savefig("check_crop.png")
    
    # print("in crop station image: ", station_coords[station_i][1]-(crop_size//2)," - " , (station_coords[station_i][1]+(crop_size//2)))

    crop = sat_image[
        :, 
        (station_coords[station_i][1]-(crop_size//2)):(station_coords[station_i][1]+(crop_size//2)), 
        (station_coords[station_i][0]-(crop_size//2)):(station_coords[station_i][0]+(crop_size//2))
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
        
        index = 0

        for index in tqdm(range(0,len(catalog),config.BATCH_SIZE)): 
        # while(index < len(catalog)):

            x = []
            y = []

            rows = catalog[ index : (index+config.BATCH_SIZE) ]
            # print(rows)

            x,y = station_from_row(args, rows, x, y)

            x = np.array(x)
            y = np.array(y)
            print("Yielding x and y (shape): ", index, x.shape,y.shape)

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

    sdl = SimpleDataLoader(args, catalog)
    
    # iterate over epochs
    for epoch in range(config.EPOCHS):
        for i,j in sdl:
            continue
            # print("Incoming x and y: ", i,j)

    print("hi i reached here")
    
    # plot_and_save_image(args,station_coords,h5_16bit_samples,prefix="h5_16")
    # plot_and_save_image(args,station_coords,ncdf_samples,prefix="ncdf")

    return SimpleDataLoader(args, catalog)

def extract_at_time(time,ctlg):
    pass

def create_data_loader():
    pass

def data_loader_main():
    args = config.init_args()
    load_dataset(args)

if __name__ == "__main__":
    data_loader_main()