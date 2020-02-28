import argparse
import datetime
import json
import os
import typing

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.data.Dataset:
    """This function should be modified in order to prepare & return your own data loader.

    Note that you can use either the netCDF or HDF5 data. Each iteration over your data loader should return a
    2-element tuple containing the tensor that should be provided to the model as input, and the target values. In
    this specific case, you will not be able to provide the latter since the dataframe contains no GHI, and we are
    only interested in predictions, not training. Therefore, you must return a placeholder (or ``None``) as the second
    tuple element.

    Reminder: the dataframe contains imagery paths for every possible timestamp requested in ``target_datetimes``.
    However, we expect that you will use some of the "past" imagery (i.e. imagery at T<=0) for any T in
    ``target_datetimes``, but you should NEVER rely on "future" imagery to generate predictions (for T>0). We
    will be inspecting data loader implementations to ensure this is the case, and those who "cheat" will be
    dramatically penalized.

    See https://github.com/mila-iqia/ift6759/tree/master/projects/project1/evaluation.md for more information.

    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
            relevant timestamp values over the test period.
        target_datetimes: a list of timestamps that your data loader should use to provide imagery for your model.
            The ordering of this list is important, as each element corresponds to a sequence of GHI values
            to predict. By definition, the GHI values must be provided for the offsets given by ``target_time_offsets``
            which are added to each timestamp (T=0) in this datetimes list.
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A ``tf.data.Dataset`` object that can be used to produce input tensors for your model. One tensor
        must correspond to one sequence of past imagery data. The tensors must be generated in the order given
        by ``target_sequences``.
    """
    ################################## MODIFY BELOW ##################################
    # WE ARE PROVIDING YOU WITH A DUMMY DATA GENERATOR FOR DEMONSTRATION PURPOSES.
    # MODIFY EVERYTHINGIN IN THIS BLOCK AS YOU SEE FIT

    # stores numpy crops in a npz file
    def store_numpy(ndarray_dict,filepath):
        os.makedirs('npz_store',exist_ok=True)
        path = os.path.splitext(os.path.basename(filepath))[0] + ".npz"
        path = os.path.join('npz_store',path)
        np.savez(path, **ndarray_dict)

    # optimized loading of crops. stores all crops per file. 
    def load_numpy(filepath):
        path = os.path.splitext(os.path.basename(filepath))[0] + ".npz"
        path = os.path.join('npz_store',path)
        if not os.path.isfile(path):
            ndarray_dict = fetch_all_samples_hdf5(filepath)
            store_numpy(ndarray_dict, filepath)
        else:
            ndarray_dict = np.load(path)
        return ndarray_dict
    
    def read_hdf5(hdf5_path):
        import h5py
        h5_data = h5py.File(hdf5_path, "r")

        return h5_data

    # modified template code from utils.py to extract all crops per station per file
    def fetch_all_samples_hdf5(h5_data_path):
        from utilities import utils
        channels = ["ch1","ch2","ch3","ch4","ch6"]

        # # return sample
        copy_last_if_missing = True
        h5_data = read_hdf5(h5_data_path)
        global_start_idx = h5_data.attrs["global_dataframe_start_idx"]
        global_end_idx = h5_data.attrs["global_dataframe_end_idx"]
        archive_lut_size = global_end_idx - global_start_idx
        global_start_time = datetime.datetime.strptime(h5_data.attrs["global_dataframe_start_time"], "%Y.%m.%d.%H%M")
        lut_timestamps = [global_start_time + idx * datetime.timedelta(minutes=15) for idx in range(archive_lut_size)]
        stations = {
          "BND": [    40.05192,    -88.37309,    230]
          ,  "TBL": [    40.12498,    -105.2368,    1689]
          ,  "DRA": [    36.62373,    -116.01947,    1007]
          ,  "FPK": [    48.30783,    -105.1017,    634]
          ,  "GWN": [    34.2547,    -89.8729,    98]
          ,  "PSU": [    40.72012,    -77.93085,    376]
          ,  "SXF": [    43.73403,    -96.62328,    473]
        } 

        stations_data = {}
        # assume lats/lons stay identical throughout all frames; just pick the first available arrays
        idx, lats, lons = 0, None, None
        while (lats is None or lons is None) and idx < archive_lut_size:
            lats, lons = utils.fetch_hdf5_sample("lat", h5_data, idx), utils.fetch_hdf5_sample("lon", h5_data, idx)
            idx += 1    
        assert lats is not None and lons is not None, "could not fetch lats/lons arrays (hdf5 might be empty)"
        for reg, coords in stations.items():
            station_coords = (np.argmin(np.abs(lats - coords[0])), np.argmin(np.abs(lons - coords[1])))
            station_data = {"coords": station_coords}
            stations_data[reg] = station_data

        raw_data = np.zeros((archive_lut_size, len(channels), 650, 1500), dtype=np.uint8)
        for channel_idx, channel_name in enumerate(channels):
            assert channel_name in h5_data, f"missing channel: {channels}"
            channel_data = [utils.fetch_hdf5_sample(channel_name, h5_data, idx) for idx in range(archive_lut_size)]
            assert all([array is None or array.shape == (650, 1500) for array in channel_data]), \
                "one of the saved channels had an expected dimension"
            last_valid_array_idx = None
            for array_idx, array in enumerate(channel_data):
                if array is None:
                    if copy_last_if_missing and last_valid_array_idx is not None:
                        raw_data[array_idx, channel_idx, :, :] = raw_data[last_valid_array_idx, channel_idx, :, :]
                    continue
                # array = (((array.astype(np.float32) - norm_min) / (norm_max - norm_min)) * 255).astype(np.uint8)
                raw_data[array_idx, channel_idx, :, :] = array
                last_valid_array_idx = array_idx
        # print("raw_data:",raw_data.shape)
        crop_size = 70
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

    # Image generator for feeding inputs to the model
    def _generator3():
        import traceback
        avg_x = np.array([0.31950477, 283.18481332, 239.19212155, 272.73521949, 254.09056291]).reshape(1,1,5)
        std_x = np.array([0.27667209, 16.24902932,  8.79865931, 20.08307892, 13.8115307]).reshape(1,1,5)

        catalog = dataframe.copy()
        catalog.loc[(catalog[station_i+'_DAYTIME']==0) & (catalog[station_i+'_CLEARSKY_GHI'].isna()), station_i+'_CLEARSKY_GHI'] = 0
        catalog[station_i+'_CLEARSKY_GHI'] = catalog[station_i+'_CLEARSKY_GHI'].interpolate(method='linear')
        catalog.loc[catalog[station_i+'_CLEARSKY_GHI'] < 0, station_i+'_CLEARSKY_GHI'] = 0 

        # for idx, row in tqdm(catalog.iterrows(),total=len(catalog)):
        for idx in tqdm.tqdm(target_datetimes):
            row = catalog[catalog.index==idx].iloc[0]
            print("HERE:",row['hdf5_8bit_offset'])
            try:
                if row.hdf5_8bit_path == "nan":
                    print("woh oh ohooo")
                    continue
                
                # 0.0001 s 
                # if not ((row.BND_DAYTIME==1) | (row.TBL_DAYTIME==1) | (row.DRA_DAYTIME==1) | (row.FPK_DAYTIME==1) | (row.GWN_DAYTIME==1) | (row.PSU_DAYTIME==1) | (row.SXF_DAYTIME==1)):
                #     continue

                # 0.05 s
                samples = load_numpy(row['hdf5_8bit_path'])
                offset_idx = row['hdf5_8bit_offset']
                
                timedelta_rows = [catalog[catalog.index==(idx+tto)] for tto in target_time_offsets]
                # ss = ["BND","TBL","DRA","FPK","GWN","PSU","SXF"]

                for station_i, value in stations.items():

                    CS_GHIs = [i[station_i + "_CLEARSKY_GHI"].values[0] for i in timedelta_rows]
                    # y = np.array(CS_GHIs)

                    # 0.05 s
                    print(samples[station_i].shape)
                    sample = samples[station_i]
                    x = sample[offset_idx].swapaxes(0,1).swapaxes(1,2)
                    x = (x - avg_x)/std_x
                    # yield ((x,CS_GHIs),CS_GHIs)
                    yield x,CS_GHIs,CS_GHIs

            except Exception as e:
    #             when an offset not in training dataset, it raises error in finding future GHIs
                print(e)
                print(traceback.format_exc())
                print("****** in except ******")
                print("Golmaal hai bhai sab golmaal hai")
                continue

    class SimpleDataLoader2(tf.data.Dataset):

        def __new__(cls, path):

            return tf.data.Dataset.from_generator(
                _generator3,
                args=([path]),
                output_types=(tf.float32,tf.float32,tf.float32),
                output_shapes=(
                   tf.TensorShape([70, 70, 5]),tf.TensorShape([4]),
                   tf.TensorShape([4])
                   )
                ).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    
    data_loader = tf.data.Dataset.from_generator(
        _generator3, (tf.float32, tf.float32, tf.float32)
    ).batch(1)

    ################################### MODIFY ABOVE ##################################

    return data_loader


def prepare_model(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
) -> tf.keras.Model:
    """This function should be modified in order to prepare & return your own prediction model.

    See https://github.com/mila-iqia/ift6759/tree/master/projects/project1/evaluation.md for more information.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A ``tf.keras.Model`` object that can be used to generate new GHI predictions given imagery tensors.
    """

    ################################### MODIFY BELOW ##################################
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, ReLU
    
    class DummyModel(tf.keras.Model):

      def __init__(self, target_time_offsets):
        super(DummyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(len(target_time_offsets), activation=tf.nn.softmax)

      def call(self, inputs):
        x = self.dense1(self.flatten(inputs))
        return self.dense2(x)

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, kernel_initializer='he_uniform')
            self.batchnorm1 = BatchNormalization()
            self.relu = ReLU()
            self.maxpool = MaxPool2D((2,2))
            self.conv2 = Conv2D(64, 3, kernel_initializer='he_uniform')
            self.batchnorm2 = BatchNormalization()
            self.conv3 = Conv2D(128, 3, kernel_initializer='he_uniform')
            self.batchnorm3 = BatchNormalization()
            self.conv4 = Conv2D(256, 3, kernel_initializer='he_uniform')
            self.batchnorm4 = BatchNormalization()
            self.conv5 = Conv2D(512, 5, kernel_initializer='he_uniform')
            self.batchnorm5 = BatchNormalization()
            self.flatten = Flatten()
            self.d1 = Dense(4, activation='linear')

        def call(self, x):
            c = x[1]
            x = x[0]
            x = self.conv1(x) # 68 x 68 x 32
            x = self.batchnorm1(x)
            x = self.relu(x)
            x = self.maxpool(x) # 34 x 34 x 32
            x = self.conv2(x) # 32 x 32 x 64
            x = self.batchnorm2(x)
            x = self.relu(x)
            x = self.maxpool(x) # 16 x 16 x 64
            x = self.conv3(x) # 14 x 14 x 128
            x = self.batchnorm3(x)
            x = self.relu(x)
            x = self.maxpool(x) # 7 x 7 x 128
            x = self.conv4(x) # 5 x 5 x 256
            x = self.batchnorm4(x)
            x = self.relu(x)
            x = self.conv5(x) # 1 x 1 x 512
            x = self.relu(x)
            x = self.batchnorm5(x)
            x = self.flatten(x)
            x = self.d1(x)
            return x+c

    model = MyModel()
    model.load_weights('checkpoints/keras_model_1582840350.1643887')

    ################################### MODIFY ABOVE ##################################

    return model


def generate_predictions(data_loader: tf.data.Dataset, model: tf.keras.Model, pred_count: int) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""
    predictions = []
    with tqdm.tqdm("generating predictions", total=pred_count) as pbar:
        for iter_idx, minibatch in enumerate(data_loader):
            assert isinstance(minibatch, tuple) and len(minibatch) >= 2, \
                "the data loader should load each minibatch as a tuple with model input(s) and target tensors"
            # remember: the minibatch should contain the input tensor(s) for the model as well as the GT (target)
            # values, but since we are not training (and the GT is unavailable), we discard the last element
            # see https://github.com/mila-iqia/ift6759/blob/master/projects/project1/datasources.md#pipeline-formatting
            if len(minibatch) == 2:  # there is only one input + groundtruth, give the model the input directly
                pred = model(minibatch[0])
            else:  # the model expects multiple inputs, give them all at once using the tuple
                pred = model(minibatch[:-1])
            if isinstance(pred, tf.Tensor):
                pred = pred.numpy()
            assert pred.ndim == 2, "prediction tensor shape should be BATCH x SEQ_LENGTH"
            predictions.append(pred)
            pbar.update(len(pred))
    return np.concatenate(predictions, axis=0)

def generate_all_predictions(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
        user_config: typing.Dict[typing.AnyStr, typing.Any],
) -> np.ndarray:
    """Generates and returns model predictions given the data prepared by a data loader."""
    # we will create one data loader per station to make sure we avoid mixups in predictions
    predictions = []
    for station_idx, station_name in enumerate(target_stations):
        # usually, we would create a single data loader for all stations, but we just want to avoid trouble...
        stations = {station_name: target_stations[station_name]}
        print(f"preparing data loader & model for station '{station_name}' ({station_idx + 1}/{len(target_stations)})")
        data_loader = prepare_dataloader(dataframe, target_datetimes, stations, target_time_offsets, user_config)
        model = prepare_model(stations, target_time_offsets, user_config)
        station_preds = generate_predictions(data_loader, model, pred_count=len(target_datetimes))
        assert len(station_preds) == len(target_datetimes), "number of predictions mismatch with requested datetimes"
        predictions.append(station_preds)
    return np.concatenate(predictions, axis=0)


def parse_gt_ghi_values(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station GHI values from the provided dataframe for the evaluation of predictions."""
    gt = []
    for station_idx, station_name in enumerate(target_stations):
        station_ghis = dataframe[station_name + "_GHI"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_ghis.index:
                    seq_vals.append(station_ghis.iloc[station_ghis.index.get_loc(index)])
                else:
                    seq_vals.append(float("nan"))
            gt.append(seq_vals)
    return np.concatenate(gt, axis=0)


def parse_nighttime_flags(
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_datetimes: typing.List[datetime.datetime],
        target_time_offsets: typing.List[datetime.timedelta],
        dataframe: pd.DataFrame,
) -> np.ndarray:
    """Parses all required station daytime flags from the provided dataframe for the masking of predictions."""
    flags = []
    for station_idx, station_name in enumerate(target_stations):
        station_flags = dataframe[station_name + "_DAYTIME"]
        for target_datetime in target_datetimes:
            seq_vals = []
            for time_offset in target_time_offsets:
                index = target_datetime + time_offset
                if index in station_flags.index:
                    seq_vals.append(station_flags.iloc[station_flags.index.get_loc(index)] > 0)
                else:
                    seq_vals.append(False)
            flags.append(seq_vals)
    return np.concatenate(flags, axis=0)


def main(
        preds_output_path: typing.AnyStr,
        admin_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
        stats_output_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    """Extracts predictions from a user model/data loader combo and saves them to a CSV file."""

    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    if "start_bound" in admin_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
    if "end_bound" in admin_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]

    target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]

    if "bypass_predictions_path" in admin_config and admin_config["bypass_predictions_path"]:
        # re-open cached output if possible (for 2nd pass eval)
        assert os.path.isfile(preds_output_path), f"invalid preds file path: {preds_output_path}"
        with open(preds_output_path, "r") as fd:
            predictions = fd.readlines()
        assert len(predictions) == len(target_datetimes) * len(target_stations), \
            "predicted ghi sequence count mistmatch wrt target datetimes x station count"
        assert len(predictions) % len(target_stations) == 0
        predictions = np.asarray([float(ghi) for p in predictions for ghi in p.split(",")])
    else:
        predictions = generate_all_predictions(target_stations, target_datetimes,
                                               target_time_offsets, dataframe, user_config)
        with open(preds_output_path, "w") as fd:
            for pred in predictions:
                fd.write(",".join([f"{v:0.03f}" for v in pred.tolist()]) + "\n")
    print(list(dataframe.columns.values))
    for s in target_stations:
        if s + "_GHI" not in dataframe:
            print(s,"not in DATAFRAME")
    if any([s + "_GHI" not in dataframe for s in target_stations]):
        print("station GHI measures missing from dataframe, skipping stats output")
        return

    assert not np.isnan(predictions).any(), "user predictions should NOT contain NaN values"
    predictions = predictions.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    gt = parse_gt_ghi_values(target_stations, target_datetimes, target_time_offsets, dataframe)
    gt = gt.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))
    day = parse_nighttime_flags(target_stations, target_datetimes, target_time_offsets, dataframe)
    day = day.reshape((len(target_stations), len(target_datetimes), len(target_time_offsets)))

    squared_errors = np.square(predictions - gt)
    stations_rmse = np.sqrt(np.nanmean(squared_errors, axis=(1, 2)))
    for station_idx, (station_name, station_rmse) in enumerate(zip(target_stations, stations_rmse)):
        print(f"station '{station_name}' RMSE = {station_rmse:.02f}")
    horizons_rmse = np.sqrt(np.nanmean(squared_errors, axis=(0, 1)))
    for horizon_idx, (horizon_offset, horizon_rmse) in enumerate(zip(target_time_offsets, horizons_rmse)):
        print(f"horizon +{horizon_offset} RMSE = {horizon_rmse:.02f}")
    overall_rmse = np.sqrt(np.nanmean(squared_errors))
    print(f"overall RMSE = {overall_rmse:.02f}")

    if stats_output_path is not None:
        # we remove nans to avoid issues in the stats comparison script, and focus on daytime predictions
        squared_errors = squared_errors[~np.isnan(gt) & day]
        with open(stats_output_path, "w") as fd:
            for err in squared_errors.reshape(-1):
                fd.write(f"{err:0.03f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_out_path", type=str,
                        help="path where the raw model predictions should be saved (for visualization purposes)")
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    parser.add_argument("-s", "--stats_output_path", type=str, default=None,
                        help="path where the prediction stats should be saved (for benchmarking)")
    args = parser.parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
        stats_output_path=args.stats_output_path,
    )
