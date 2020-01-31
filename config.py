import argparse,os

ROOT_DATA_PATH = '/project/cq-training-1/project1/data'
RAW_DATA_CATALOG = 'catalog.helios.public.20100101-20160101.pkl'
TEST_DATA_CATALOG = 'dummy_test_catalog.pkl'

# SURFRAD Station codes
STATION_CODES = ['BND','TBL','DRA','FPK','GWN','PSU','SXF']

def init_args():
	parser = argparse.ArgumentParser(description='Process arguments for processing/training/testing')
	parser.add_argument('--mode', help='train/test/prepro')
	parser.add_argument('--image_data', choices=['hdf5v7_8bit','hdf5v5_16bit'], help='choose either hdf5 8bit or 16bit')
	parser.add_argument('--data_catalog',help='catalog file (dataframe) containing image path, ghi and other metadata')
	
	args = parser.parse_args()
	return args