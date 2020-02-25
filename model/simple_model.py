import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Input, Dense, MaxPool2D, BatchNormalization, ReLU
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from utilities import config
from utilities.dataloader2 import SequenceDataLoaderMemChunks
from utilities.dataloader_simple import SimpleDataLoader
from utilities.utility import load_catalog
import pdb

class MyModel(Model):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.input_0 = Input(shape=(70,70,5))
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
        self.d1 = Dense(1, activation='linear')

    def call(self, x):
        # x = x[0]
        x = self.input_0(x)
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
        return x# Create an instance of the model

if __name__ == "__main__":
    args = config.init_args()

    model = MyModel(args)
    optimizer = Adam(learning_rate=args.lr)
    rmse = RootMeanSquaredError()
    model.compile(optimizer=optimizer, 
                  loss='mean_squared_error',
                  metrics=[rmse])
    # print(model.summary())
    catalog_train = load_catalog(args.data_catalog_path)
    catalog_val = load_catalog(args.val_catalog_path)
    catalog_test = load_catalog(args.test_catalog_path)
    sdl_train = SimpleDataLoader(args, catalog_train).prefetch(tf.data.experimental.AUTOTUNE).batch(args.batch_size)
    sdl_val = SimpleDataLoader(args, catalog_val).prefetch(tf.data.experimental.AUTOTUNE).batch(args.batch_size)
    model.fit(
        sdl_train,
        steps_per_epoch=10000,
        epochs=5, 
        validation_data=sdl_val,
        validation_steps=1000
    )