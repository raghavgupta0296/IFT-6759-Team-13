import datetime
from functools import lru_cache
import pickle
import pdb
from time import sleep
import typing

import cv2
import h5py, h5netcdf
from line_profiler import LineProfiler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import config
from utilities import utils

"""
Generates a file name
Args:
    length:  length of the string 
Returns:
    string of size 'length' containing random characters
"""
def get_file_name(length = 15):
    return binascii.b2a_hex(os.urandom(length)).decode('ascii')

# loads the pickle dataframe containing data paths and targets information
def load_catalog(args):
    f = open(args.data_catalog_path,"rb")
    dataset = pickle.load(f)
    f.close()
    return dataset

# loads an hdf5 file
@lru_cache(maxsize=10)
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
    sample = [utils.fetch_hdf5_sample(ch, h5_data_handle, hdf5_offset) for ch in channels]
    return sample

def fetch_all_samples_hdf5(args,h5_data_path,dataframe_path=None):
    channels = args.channels

    # sample = [utils.fetch_hdf5_sample(ch, h5_data_handle, hdf5_offset) for ch in channels]
    # # return sample
    copy_last_if_missing = True
    h5_data = read_hdf5(h5_data_path)
    global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
    global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
    archive_lut_size = global_end_idx - global_start_idx
    global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
    lut_timestamps = [global_start_time + idx * datetime.timedelta(minutes=15) for idx in range(archive_lut_size)]
    stations = args.station_data
    stations_data = {}
    # df = pd.read_pickle(dataframe_path)
    # assume lats/lons stay identical throughout all frames; just pick the first available arrays
    idx, lats, lons = 0, None, None
    while (lats is None or lons is None) and idx < archive_lut_size:
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, idx), utils.fetch_hdf5_sample("lon", h5_data, idx)
        idx += 1    
    assert lats is not None and lons is not None, "could not fetch lats/lons arrays (hdf5 might be empty)"
    for reg, coords in stations.items():
        station_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
        station_data = {"coords": station_coords}
        # if dataframe_path:
        # station_data["ghi"] = [df.at[pd.Timestamp(t), reg + "_GHI"] for t in lut_timestamps]
        # station_data["csky"] = [df.at[pd.Timestamp(t), reg + "_CLEARSKY_GHI"] for t in lut_timestamps]
        # station_data["daytime"] = [df.at[pd.Timestamp(t), reg + "_DAYTIME"] for t in lut_timestamps]
        stations_data[reg] = station_data

    raw_data = np.zeros((archive_lut_size, len(channels), 650, 1500), dtype=np.uint8)
    for channel_idx, channel_name in enumerate(channels):
        assert channel_name in h5_data, f"missing channel: {channels}"
        norm_min = h5_data[channel_name].attrs.get("orig_min", None)
        norm_max = h5_data[channel_name].attrs.get("orig_max", None)
        channel_data = [utils.fetch_hdf5_sample(channel_name, h5_data, idx) for idx in range(archive_lut_size)]
        assert all([array is None or array.shape == (650, 1500) for array in channel_data]), \
            "one of the saved channels had an expected dimension"
        last_valid_array_idx = None
        for array_idx, array in enumerate(channel_data):
            if array is None:
                if copy_last_if_missing and last_valid_array_idx is not None:
                    raw_data[array_idx, channel_idx, :, :] = raw_data[last_valid_array_idx, channel_idx, :, :]
                continue
            array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
            # array = cv.applyColorMap(array, cv.COLORMAP_BONE)
            # for station_idx, (station_name, station) in enumerate(stations_data.items()):
            #     station_color = get_label_color_mapping(station_idx + 1).tolist()[::-1]
            #     array = cv.circle(array, station["coords"][::-1], radius=9, color=station_color, thickness=-1)
            # raw_data[array_idx, channel_idx, :, :] = cv.flip(array, 0)
            raw_data[array_idx, channel_idx, :, :] = array
            last_valid_array_idx = array_idx
    print("raw_data:",raw_data.shape)
    

    crop_size = args.crop_size
    station_crops = {}
    for station_name, station in stations_data.items():
        # array = cv.circle(array, station["coords"][::-1], radius=9, color=station_color, thickness=-1)
        station_coords = station["coords"]
        margin = crop_size//2
        lat_mid = station_coords[0]
        lon_mid = station_coords[1]
        crop = raw_data[
            :, :,
            lat_mid-margin:lat_mid+margin, 
            lon_mid-margin:lon_mid+margin, 
        ]
        station_crops[station_name] = crop
    return station_crops

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
        lat, lon = utils.fetch_hdf5_sample("lat", h5_data, idx), utils.fetch_hdf5_sample("lon", h5_data, idx)
    return lat, lon

def get_hdf5_attributes(
        dataset_name: str,
        reader: h5py.File,
        sample_idx: int,
    ) -> typing.Any:
    """
    Displays list of attributes
    Args:
        dataset_name: name of the HDF5 dataset to fetch the sample from using the reader. In the context of
            the GHI prediction project, this may be for example an imagery channel name (e.g. "ch1").
        reader: an HDF5 archive reader obtained via ``h5py.File(...)`` which can be used for dataset indexing.
        sample_idx: the integer index (or offset) that corresponds to the position of the sample in the dataset.

    Returns:
        The attributes keys in shape of str-list
    """
    dataset_lut_name = dataset_name + "_LUT"
    if dataset_lut_name in reader:
        sample_idx = reader[dataset_lut_name][sample_idx]
        if sample_idx == -1:
            return None  # unavailable
    dataset = reader[dataset_name]
    return dataset

def get_hdf5_fields(reader: h5py.File) -> typing.Any:
    fields_names = []
    for dataset_lut_name in reader:
        fields_names.append(dataset_lut_name)
    return fields_names