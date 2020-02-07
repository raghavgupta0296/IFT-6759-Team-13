import argparse
import json
import os

ROOT_DATA_PATH = '/project/cq-training-1/project1/data'
RAW_DATA_CATALOG = 'catalog.helios.public.20100101-20160101.pkl'
TEST_DATA_CATALOG = 'dummy_test_catalog.pkl'

# SURFRAD Station Info
# Format: { Station Code : [Latitude,Longitude,Elevation] }

CHANNELS = ["ch1","ch2","ch3","ch4","ch6"]
STATION_INFO = json.loads(open('SURFRAD.json').read())
<<<<<<< HEAD
=======
STATION_NAMES = list(STATION_INFO.keys())

# HYPERPARAMS
CROP_SIZE = 70 # even number
EPOCHS = 1
BATCH_SIZE = 64
>>>>>>> 8c64ae904973e1de9f22e0542a989c97abc86aaa

def init_args():
	parser = argparse.ArgumentParser(description='Process arguments for processing/training/testing')
	parser.add_argument('--mode', help='train/test/prepro')
	parser.add_argument('--image_data', choices=['hdf5v7_8bit','hdf5v5_16bit','netcdf'], default='hdf5v5_16bit',
		help='choose either hdf5 8bit or 16bit')

	data_catalog = os.path.join(ROOT_DATA_PATH,RAW_DATA_CATALOG)
	parser.add_argument('--data_catalog_path',default=data_catalog,help='catalog absolute file path (dataframe) which contains the image paths, ghi and other metadata')
	parser.add_argument('--station_data',default=STATION_INFO,help='catalog absolute file path (dataframe) which contains the image paths, ghi and other metadata')
	parser.add_argument('--channels',default=CHANNELS,help='catalog absolute file path (dataframe) which contains the image paths, ghi and other metadata')

<<<<<<< HEAD
=======
	parser.add_argument('--crop_size',default=CROP_SIZE,help='window size for cropping station image from satellite image.')    
    
>>>>>>> 8c64ae904973e1de9f22e0542a989c97abc86aaa
	args = parser.parse_args()

	# args["station_data"] = STATION_INFO
	# args["channels"] = CHANNELS
	return args