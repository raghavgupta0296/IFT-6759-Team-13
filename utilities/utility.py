import cv2
import pickle
import h5py, h5netcdf

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utilities.utils import fetch_hdf5_sample
from time import sleep

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

"""
Function creating dummy image. Used for as well for experiements as for generating noise/ replacing missing images
Args:
Returns:
    Blank image
"""
def create_dummy_image():
    img = np.zeros([70,70,5],dtype=np.uint8)
    #plt.imshow(img[:,:,0], cmap='gray')
    return img

"""
Dummy cropping function used for experiments. To be replaced by a valid one
Args:
    hdf5_8bit_path:  path to file
    stations_coordinates: dictionary containing stations coordinates
    offsets: list of valid offsets
Returns:
    Dictionary containing stations, offsets and corresponding images
"""
def dummy_crop_image(hdf5_8bit_path, stations_coordinates, offsets):
    global_dictionary = {}
    # Simulating access to files
    sleep(1.14)
    station_names = ['BND', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF', 'TBL']
    for s in station_names:
        dic = {}
        for o in offsets:
            dic[o] = create_dummy_image()
        global_dictionary[s] = dic
    return global_dictionary


# maps a physical co-ordinate to image pixel
def map_coord_to_pixel(coord,min_coord,res):
    x = int(abs(min_coord - coord)/res)
    return x

# extract images of all the 5 channels given offset and data handle
def fetch_channel_samples(args,h5_data_handle,hdf5_offset):
    channels = args.channels
    sample = [fetch_hdf5_sample(ch, h5_data_handle, hdf5_offset) for ch in channels]
    return sample

# saves images of 5 channels with plotted mapped co-ordinates
def plot_and_save_image(args, station_coords, samples, prefix="0"):
    cmap='gray'
    for sample,ch in zip(samples, args.channels):
        plt.imshow(sample,origin='lower',cmap=cmap)
        plt.scatter(station_coords[:,0], station_coords[:,1])
        plt.savefig("sample_outputs/%s_%s.png"%(prefix, ch))


# Extracts the Latitude and Longitude  from the hdf5 data
def get_lon_lat_from_hdf5(h5_data):
    global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
    global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
    archive_lut_size = global_end_idx - global_start_idx
    idx, lat, lon = 0, None, None
    if idx < archive_lut_size:
        lat, lon = fetch_hdf5_sample("lat", h5_data, idx), fetch_hdf5_sample("lon", h5_data, idx)
    return lat, lon