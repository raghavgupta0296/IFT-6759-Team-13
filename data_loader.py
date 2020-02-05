import pickle

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py, h5netcdf

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
    return pp_dataset

# crop station image from satellite image of size CROP_SIZE
def crop_station_image(station_i,sat_image,station_coords):

    # R! check  crop correct positions? and also if lower origin needs to be taken before manual cropping
    
    crop_size = config.CROP_SIZE

    fig,ax = plt.subplots(1)
    ax.imshow(sat_image[0], cmap='bone')
    rect = Rectangle((station_coords[station_i][0]-(crop_size//2),station_coords[station_i][1]-(crop_size//2)),crop_size,crop_size,linewidth=1,fill=True,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.scatter(station_coords[station_i][0],station_coords[station_i][1])
    plt.savefig("check_crop.png")
    
    crop = sat_image[
        :, 
        (station_coords[station_i][1]-(crop_size//2)):(station_coords[station_i][1]+(crop_size//2)), 
        (station_coords[station_i][0]-(crop_size//2)):(station_coords[station_i][0]+(crop_size//2))
        ]

    if crop.shape!=(5,crop_size,crop_size):
        print("crop channels shape:", station_i, [crop[i].shape for i in range(len(crop))])
    
    plt.imshow(crop[0], cmap='bone')
    plt.savefig("check_cropped.png")

    return crop

# input output pairs for dataset
def create_io_pairs(row,sat_image,station_coords):
    station_ims = []
    station_ghis = []
    
    for station_i in config.STATION_NAMES:
    # station_i = 'FPK'
        print(station_i)
        station_ghis.append(row[station_i+"_GHI"])
        station_ims.append(crop_station_image(station_i,sat_image,station_coords))
        print(np.array(station_ims).shape)
    
    station_ims = np.array(station_ims)
    station_ghis = np.array(station_ghis)
    print("station ims shape: ", station_ims.shape)
    print("station ghi shape: ", station_ghis.shape, station_ghis)
    return station_ims, station_ghis

def iterate_dataset():
    # R! vectorize this by using iloc instead of iterrows?  
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

        station_coords = {}
        # R! reading from dictionary can lead to random order of key pairs, 
        # storing datatype changed from array to dictionary
        for sta, (lat,lon,elev) in args.station_data.items():
            # y = row data (longitude: changes across rows i.e. vertically)
            # x = column data (latitude: changes across columns i.e horizontally)
            x,y = [map_coord_to_pixel(lat,lat_min,lat_res),map_coord_to_pixel(lon,lon_min,lon_res)]
            station_coords[sta] = [y,x] 

        # reads h5 and ncdf samples
        h5_16bit_samples = fetch_channel_samples(args,h5_data,row['hdf5_16bit_offset'])
        ncdf_samples = [ncdf_data.variables[ch][0] for ch in args.channels]

        # R! question: -ve large values in ncdf_samples?
        # print(ncdf_samples)

        h5_16bit_samples = np.array(h5_16bit_samples)
        print(h5_16bit_samples)
        print(type(h5_16bit_samples))
        print(h5_16bit_samples.shape)

        io = create_io_pairs(row,h5_16bit_samples,station_coords)
        yield io

# loads dataset and iterates over dataframe rows as well as hdf5 and nc files for processing
def load_dataset(args):
    catalog = load_catalog(args)
    catalog = pre_process(catalog)
    
    # data_generator = iterate_dataset(args,catalog)
    # print(data_generator.next())

    tf_set = tf.data.Dataset.from_generator(iterate_dataset, (tf.float32,tf.float32))
    print(tf_set)

    print("hi i reached here")
    
    # plot_and_save_image(args,station_coords,h5_16bit_samples,prefix="h5_16")
    # plot_and_save_image(args,station_coords,ncdf_samples,prefix="ncdf")

def extract_at_time(time,ctlg):
    pass

def create_data_loader():
    pass

def data_loader_main():
    args = config.init_args()
    load_dataset(args)

if __name__ == "__main__":
    data_loader_main()