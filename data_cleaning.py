import pickle
import pandas as pd

pd.set_option('display.max_columns',None)

stations = ["BND","TBL","DRA","FPK","GWN","PSU","SXF"]

with open("/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl","rb") as f:
    dataset = pickle.load(f)

# no nighttime rows (include rows even if day at one station)
dataset_day = dataset[(dataset.BND_DAYTIME==1) | (dataset.TBL_DAYTIME==1) | (dataset.DRA_DAYTIME==1) | (dataset.FPK_DAYTIME==1) | (dataset.GWN_DAYTIME==1) | (dataset.PSU_DAYTIME==1) | (dataset.SXF_DAYTIME==1)]

# daytime and no ncdf paths
dataset_day_av = dataset_day[dataset_day.ncdf_path!="nan"]

# melted data
clean_data = dataset_day_av.reset_index()

for station_i in stations:
    clean_data[station_i] = clean_data[station_i+"_DAYTIME"].astype('str') + "~=~" + clean_data[station_i+"_CLEARSKY_GHI"].astype('str') + "~=~" + clean_data[station_i+"_CLOUDINESS"].astype('str') + "~=~" + clean_data[station_i+"_GHI"].astype('str')

clean_data = clean_data.drop(columns=['BND_DAYTIME',
       'BND_CLEARSKY_GHI', 'BND_CLOUDINESS', 'BND_GHI', 'TBL_DAYTIME',
       'TBL_CLEARSKY_GHI', 'TBL_CLOUDINESS', 'TBL_GHI', 'DRA_DAYTIME',
       'DRA_CLEARSKY_GHI', 'DRA_CLOUDINESS', 'DRA_GHI', 'FPK_DAYTIME',
       'FPK_CLEARSKY_GHI', 'FPK_CLOUDINESS', 'FPK_GHI', 'GWN_DAYTIME',
       'GWN_CLEARSKY_GHI', 'GWN_CLOUDINESS', 'GWN_GHI', 'PSU_DAYTIME',
       'PSU_CLEARSKY_GHI', 'PSU_CLOUDINESS', 'PSU_GHI', 'SXF_DAYTIME',
       'SXF_CLEARSKY_GHI', 'SXF_CLOUDINESS', 'SXF_GHI'])

clean_data = clean_data.melt(id_vars=['iso-datetime', 'ncdf_path', 'hdf5_8bit_path', 'hdf5_8bit_offset',
       'hdf5_16bit_path', 'hdf5_16bit_offset'], value_vars=['BND', 'TBL', 'DRA', 'FPK',
       'GWN', 'PSU', 'SXF'], var_name='station', value_name='station_info')

split_cols = clean_data.station_info.str.split("~=~",expand=True)
clean_data['DAYTIME'] = split_cols[0]
clean_data['CLEARSKY_GHI'] = split_cols[1]
clean_data['CLOUDINESS'] = split_cols[2]
clean_data['GHI'] = split_cols[3]

clean_data = clean_data.drop(columns=['station_info'])

clean_data.DAYTIME = clean_data.DAYTIME.astype('float')
clean_data.CLEARSKY_GHI = clean_data.CLEARSKY_GHI.astype('float')
clean_data.GHI = clean_data.GHI.astype('float')

# with open("melted_df","rb") as f:
#     clean_data = pickle.load(f)

clean_data = clean_data[clean_data.DAYTIME==1]

clean_data = clean_data[clean_data.GHI.isna()==False]

clean_data.loc[clean_data.GHI<0,'GHI'] = 0

clean_data = clean_data.reset_index(drop=True)

# with open("clean_df","wb") as f:
#     pickle.dump(clean_data,f)

print(clean_data)
