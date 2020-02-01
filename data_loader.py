import pickle

import cv2
import matplotlib.pyplot as plt
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
	cmap='gray'
	for sample,ch in zip(samples,args.channels):
		plt.imshow(sample,origin='lower',cmap=cmap)
		plt.scatter(station_coords[:,0],station_coords[:,1])
		plt.savefig("sample_outputs/%s_%s.png"%(prefix,ch))

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
			x,y = [map_coord_to_pixel(lat,lat_min,lat_res),map_coord_to_pixel(lon,lon_min,lon_res)]
			station_coords.append([y,x]) 
		station_coords = np.array(station_coords)

		# reads h5 and ncdf samples
		h5_16bit_samples = fetch_channel_samples(args,h5_data,row['hdf5_16bit_offset'])
		ncdf_samples = [ncdf_data.variables[ch][0] for ch in args.channels]

		plot_and_save_image(args,station_coords,h5_16bit_samples,prefix="h5_16")
		plot_and_save_image(args,station_coords,ncdf_samples,prefix="ncdf")
		break

def extract_at_time(time,ctlg):
	pass

def create_data_loader():
	pass

def data_loader_main():
	args = config.init_args()
	load_dataset(args)

if __name__ == "__main__":
	data_loader_main()