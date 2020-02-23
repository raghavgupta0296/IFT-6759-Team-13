import argparse
import json
import os

ROOT_DATA_PATH = '/project/cq-training-1/project1/data'
RAW_DATA_CATALOG = 'catalog.helios.public.20100101-20160101.pkl'
TEST_DATA_CATALOG = 'dummy_test_catalog.pkl'

# SURFRAD Station Info
# Format: { Station Code : [Latitude,Longitude,Elevation] }

CHANNELS = ["ch1","ch2","ch3","ch4","ch6"]
STATION_INFO = json.loads(open('config/SURFRAD.json').read())
STATION_NAMES = list(STATION_INFO.keys())

# HYPERPARAMS
CROP_SIZE = 70 # even number
EPOCHS = 1
BATCH_SIZE = 64

def init_args():
	parser = argparse.ArgumentParser(description='Process arguments for processing/training/testing')
	parser.add_argument('--mode', help='train/test/prepro')
	parser.add_argument('--image_data', choices=['hdf5v7_8bit','hdf5v5_16bit','netcdf'], default='hdf5v7_8bit',
		help='choose either hdf5 8bit or 16bit')

	data_catalog = os.path.join(ROOT_DATA_PATH,RAW_DATA_CATALOG)
	parser.add_argument('--data_catalog_path',default=data_catalog,help='train catalog absolute file path (dataframe) which contains the image paths, ghi and other metadata')
	parser.add_argument('--val_catalog_path',default='valid_df',help='validation catalog absolute file path (dataframe) which contains the image paths, ghi and other metadata')
	parser.add_argument('--test_catalog_path',default='test_df',help='test catalog absolute file path (dataframe) which contains the image paths, ghi and other metadata')
	parser.add_argument('--station_data',default=STATION_INFO,help='catalog absolute file path (dataframe) which contains the image paths, ghi and other metadata')
	parser.add_argument('--channels',default=CHANNELS,help='list of channels to extract')

	parser.add_argument('--train_steps',default=100000,type=int,help='number of training examples.')
	parser.add_argument('--val_steps',default=40000,type=int,help='number of validation examples.')
	parser.add_argument('--lr',default=0.005,type=float,help='lr of the model.')
	parser.add_argument('--crop_size',default=CROP_SIZE,help='window size for cropping station image from satellite image.')    
	parser.add_argument('--batch_size',default=BATCH_SIZE,help='batch size for training data.')    
	parser.add_argument('--k_sequences',default=1,help='how many image sequences in the past to take')
	parser.add_argument('--img_sequence_step',default=4,help='step size for k_sequences. in offset unit terms')    
	parser.add_argument('--future_ghis',default=1,help='''how many future GHIs to predict apart from T0.
		Future GHIs predicted: [+1hr] if 1, [+1hr,+3hr] if 2, [+1hr,+3hr,+6hr] if 3,''')    
	parser.add_argument('-d','--debug',action="store_true",help='debugs on a limited amount of data and model capacity')
	parser.add_argument('-e','--epochs',default=EPOCHS,type=int,help='number of passes on the full dataset to train the model on')

	args = parser.parse_args()

	return args
