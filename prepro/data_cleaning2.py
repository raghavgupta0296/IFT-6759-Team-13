import pickle
import pandas as pd
stations = ["BND","TBL","DRA","FPK","GWN","PSU","SXF"]
def clean_dataset(df):
	clean_data = df.reset_index()
	for station_i in stations:
		clean_data[station_i] = clean_data[station_i+"_DAYTIME"].astype('str') + "~=~" + clean_data[station_i+"_CLEARSKY_GHI"].astype('str') 
	for station_i in stations:
		clean_data = clean_data.drop(columns=[station_i+'_DAYTIME',station_i+'_CLEARSKY_GHI'])
	clean_data = clean_data.melt(
		id_vars=['iso-datetime', 'ncdf_path', 'hdf5_8bit_path', 'hdf5_8bit_offset', 'hdf5_16bit_path', 'hdf5_16bit_offset'], 
		   value_vars=stations, 
		   var_name='station', 
		   value_name='station_info'
		   )
	split_cols = clean_data.station_info.str.split("~=~",expand=True)
	clean_data['DAYTIME'] = split_cols[0]
	clean_data['CLEARSKY_GHI'] = split_cols[1]
	clean_data = clean_data.drop(columns=['station_info'])
	clean_data.DAYTIME = clean_data.DAYTIME.astype('float')
	clean_data.CLEARSKY_GHI = clean_data.CLEARSKY_GHI.astype('float')
	clean_data = clean_data.reset_index(drop=True)
	return clean_data

with open("/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl","rb") as f:
	dataset = pickle.load(f)

for station_i in stations:
	# night and GHI nan -> 0 
	dataset.loc[(dataset[station_i+'_DAYTIME']==0) & (dataset[station_i+'_GHI'].isna()), station_i+'_GHI'] = 0
	dataset.loc[(dataset[station_i+'_DAYTIME']==0) & (dataset[station_i+'_CLEARSKY_GHI'].isna()), station_i+'_CLEARSKY_GHI'] = 0
	# GHI nan's linear interpolation
	dataset[station_i+'_GHI'] = dataset[station_i+'_GHI'].interpolate(method='linear')
	dataset[station_i+'_CLEARSKY_GHI'] = dataset[station_i+'_CLEARSKY_GHI'].interpolate(method='linear')
	# ghis <0 = 0
	dataset.loc[dataset[station_i+'_GHI'] < 0, station_i+'_GHI'] = 0
	dataset.loc[dataset[station_i+'_CLEARSKY_GHI'] < 0, station_i+'_CLEARSKY_GHI'] = 0

clean_data = clean_dataset(dataset)# Splitting of data
train_split = 0.79
valid_split = 0.20
# test_split = 0.1
train_idxs = []
valid_idxs = []
test_idxs = []
clean_data_stations = clean_data.groupby(by='station')
for i,j in clean_data_stations:
	a = int(len(j)*train_split)
	b = a + int(len(j)*valid_split)
	train_idxs = train_idxs+list(j[:a].index)
	valid_idxs = valid_idxs+list(j[a:b].index)
	test_idxs = test_idxs+list(j[b:].index)
	
print("Split: ", len(train_idxs), len(valid_idxs), len(test_idxs))
train_data = clean_data.iloc[train_idxs].reset_index(drop=True)
valid_data = clean_data.iloc[valid_idxs].reset_index(drop=True)
test_data = clean_data.iloc[test_idxs].reset_index(drop=True)
with open("train_df","wb") as f:
	pickle.dump(train_data,f)
with open("valid_df","wb") as f:
	pickle.dump(valid_data,f)
with open("test_df","wb") as f:
	pickle.dump(test_data,f)


