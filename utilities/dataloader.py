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
    pp_dataset = dataset[(dataset.BND_DAYTIME == 1) | (dataset.TBL_DAYTIME == 1) | (dataset.DRA_DAYTIME == 1) | (
                dataset.FPK_DAYTIME == 1) | (dataset.GWN_DAYTIME == 1) | (dataset.PSU_DAYTIME == 1) | (
                                     dataset.SXF_DAYTIME == 1)]
    # no empty path images
    pp_dataset = pp_dataset[pp_dataset.ncdf_path != "nan"]
    return pp_dataset


# crop station image from satellite image of size CROP_SIZE
def crop_station_image(station_i, sat_image, station_coords):
    # R! check  crop correct positions? and also if lower origin needs to be taken before manual cropping

    crop_size = CROP_SIZE

    fig, ax = plt.subplots(1)
    ax.imshow(sat_image[0], cmap='bone')
    rect = Rectangle((station_coords[station_i][0] - (crop_size // 2), station_coords[station_i][1] - (crop_size // 2)),
                     crop_size, crop_size, linewidth=1, fill=True, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.scatter(station_coords[station_i][0], station_coords[station_i][1])
    plt.savefig("check_crop.png")

    crop = sat_image[
           :,
           (station_coords[station_i][1] - (crop_size // 2)):(station_coords[station_i][1] + (crop_size // 2)),
           (station_coords[station_i][0] - (crop_size // 2)):(station_coords[station_i][0] + (crop_size // 2))
           ]

    if crop.shape != (5, crop_size, crop_size):
        print("crop channels shape:", station_i, [crop[i].shape for i in range(len(crop))])

    plt.imshow(crop[0], cmap='bone')
    plt.savefig("check_cropped.png")

    return crop


# input output pairs for dataset
def create_io_pairs(row, sat_image, station_coords):
    station_ims = []
    station_ghis = []

    for station_i in STATION_NAMES:
        # station_i = 'FPK'
        print(station_i)
        station_ghis.append(row[station_i + "_GHI"])
        station_ims.append(crop_station_image(station_i, sat_image, station_coords))
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
        for sta, (lat, lon, elev) in args.station_data.items():
            # y = row data (longitude: changes across rows i.e. vertically)
            # x = column data (latitude: changes across columns i.e horizontally)
            x, y = [map_coord_to_pixel(lat, lat_min, lat_res), map_coord_to_pixel(lon, lon_min, lon_res)]
            station_coords[sta] = [y, x]

            # reads h5 and ncdf samples
        h5_16bit_samples = fetch_channel_samples(args, h5_data, row['hdf5_16bit_offset'])
        ncdf_samples = [ncdf_data.variables[ch][0] for ch in args.channels]

        # R! question: -ve large values in ncdf_samples?
        # print(ncdf_samples)

        h5_16bit_samples = np.array(h5_16bit_samples)
        print(h5_16bit_samples)
        print(type(h5_16bit_samples))
        print(h5_16bit_samples.shape)

        io = create_io_pairs(row, h5_16bit_samples, station_coords)
        yield io


# loads dataset and iterates over dataframe rows as well as hdf5 and nc files for processing
def load_dataset(args):
    catalog = load_catalog(args)
    catalog = pre_process(catalog)

    # data_generator = iterate_dataset(args,catalog)
    # print(data_generator.next())

    tf_set = tf.data.Dataset.from_generator(iterate_dataset, (tf.float32, tf.float32))
    print(tf_set)
    print("hi i reached here")

    # plot_and_save_image(args,station_coords,h5_16bit_samples,prefix="h5_16")
    # plot_and_save_image(args,station_coords,ncdf_samples,prefix="ncdf")


def extract_at_time(time, ctlg):
    pass


def create_data_loader():
    pass


def data_loader_main():
    args = init_args()
    load_dataset(args)


if __name__ == "__main__":
    data_loader_main()