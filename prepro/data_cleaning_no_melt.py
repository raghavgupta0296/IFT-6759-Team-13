import pickle
import pandas as pd

with open("/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl","rb") as f:
    dataset = pickle.load(f)
    
stations = ["BND","TBL","DRA","FPK","GWN","PSU","SXF"]

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
    
# dataset = dataset[dataset.ncdf_path!="nan"]
# dataset = dataset[(dataset.BND_DAYTIME==1) | (dataset.TBL_DAYTIME==1) | (dataset.DRA_DAYTIME==1) | (dataset.FPK_DAYTIME==1) | (dataset.GWN_DAYTIME==1) | (dataset.PSU_DAYTIME==1) | (dataset.SXF_DAYTIME==1)]

train_idx = int(len(dataset)*0.79)
valid_idx = int(train_idx + len(dataset)*0.20)

train_dataset = dataset.iloc[:train_idx] # 1605 days
valid_dataset = dataset.iloc[train_idx:valid_idx] # 416 days
test_dataset = dataset.iloc[valid_idx:] #26 days

with open("train_df","wb") as f:
    pickle.dump(train_dataset,f)
    
with open("valid_df","wb") as f:
    pickle.dump(valid_dataset,f)
    
with open("test_df","wb") as f:
    pickle.dump(test_dataset,f)
    