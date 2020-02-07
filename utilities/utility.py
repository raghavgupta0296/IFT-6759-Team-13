import pickle
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py, h5netcdf

import matplotlib.pyplot as plt
from utilities.utils import fetch_hdf5_sample


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