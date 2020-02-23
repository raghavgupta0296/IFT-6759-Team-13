import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import wandb, pickle
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers, Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from tensorflow.keras import optimizers

from utilities.dataloader import SequenceDataLoader,SequenceDataLoaderMemChunks
from utilities import config

def get_3d_model():
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(2,3,3), activation='relu', input_shape=(2,70,70,5)))
    model.add(MaxPooling3D(pool_size=(1,3,3)))
    model.add(Conv3D(16, kernel_size=(1,5,5), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,3,3)))
    model.add(Conv3D(10, kernel_size=(1,5,5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10,activation='linear'))
    model.add(Dense(2,activation='linear'))
    return model

def get_3d_model_new():
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(2,5,5), activation='relu', input_shape=(2,70,70,5)))
    model.add(MaxPooling3D(pool_size=(1,3,3)))
    model.add(Conv3D(64, kernel_size=(1,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,3,3)))
    model.add(Conv3D(128, kernel_size=(1,3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10,activation='linear'))
    model.add(Dense(2,activation='linear'))
    return model

# class MyModel(Model):
#   def __init__(self):
#       super(MyModel, self).__init__()
#       self.conv1 = Conv2D(32, 3, activation='None')
#       self.batchnorm1 = BatchNormalization()
#       self.relu = ReLU()
#       self.maxpool = MaxPool2D((2,2))
#       self.conv2 = Conv2D(64, 3, activation='None')
#       self.batchnorm2 = BatchNormalization()
#       self.conv3 = Conv2D(128, 3, activation='None')
#       self.batchnorm3 = BatchNormalization()
#       self.conv4 = Conv2D(256, 3, activation='None')
#       self.batchnorm4 = BatchNormalization()
#       self.conv5 = Conv2D(512, 5, activation='None')
#       # self.batchnorm5 = BatchNormalization()
#       self.flatten = Flatten()
#       self.d1 = Dense(1, activation='linear')

#   def call(self, x):
#       x = self.conv1(x) # 68 x 68 x 32
#       x = self.batchnorm1(x)
#       x = self.relu(x)
#       x = self.maxpool(x) # 34 x 34 x 32
#       x = self.conv2(x) # 32 x 32 x 64
#       x = self.batchnorm2(x)
#       x = self.relu(x)
#       x = self.maxpool(x) # 16 x 16 x 64
#       x = self.conv3(x) # 14 x 14 x 128
#       x = self.batchnorm3(x)
#       x = self.relu(x)
#       x = self.maxpool(x) # 7 x 7 x 128
#       x = self.conv4(x) # 5 x 5 x 256
#       x = self.batchnorm4(x)
#       x = self.relu(x)
#       x = self.conv5(x) # 1 x 1 x 512
#       x = self.relu(x)
#       # x = self.batchnorm5(x)
#       x = self.flatten(x)
#       x = self.d1(x)
#       return x


# # Create an instance of the model
# model = MyModel()

# loss_object = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# valid_loss = tf.keras.metrics.Mean(name='valid_loss')


# @tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        # for i in range(10):
        #   print(predictions[i].numpy(), labels[i].numpy())
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    print("\n Loss: ", loss.numpy())


# @tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    print("\n Valid Loss: ", v_loss)

if __name__ == "__main__":
    from utilities.utility import load_catalog
    print(tf.__version__)
    # wandb.init(project="advanced-projects-in-ml-1")
    
    args = config.init_args()

    catalog_train = load_catalog(args.data_catalog_path)
    catalog_val = load_catalog(args.val_catalog_path)
    catalog_test = load_catalog(args.test_catalog_path)

    model = get_3d_model_new()
    optimizer = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    print(model.summary())
    # exit()
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    for epoch in range(args.epochs):
        print("EPOCH ", epoch)
        
        sdl_train = SequenceDataLoaderMemChunks(args, catalog_train).batch(64)
        sdl_val = SequenceDataLoaderMemChunks(args, catalog_val).batch(64)
        # sdl_test = SequenceDataLoaderMemChunks(args, catalog_test)

        history = model.fit_generator(generator=sdl_train,
                                validation_data=sdl_val,
                                workers=4)
        os.makedirs('history',exist_ok=True)
        with open('history/epoch_dict_%d'%epoch, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        # use_multiprocessing=True,
