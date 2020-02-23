import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import wandb, pickle, os
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

def get_3d_model_new(args):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(args.k_sequences+1,5,5),
        padding='same',activation='relu', input_shape=(args.k_sequences+1,70,70,5)))
    model.add(MaxPooling3D(pool_size=(1,3,3)))
    model.add(Conv3D(64, kernel_size=(1,3,3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,3,3)))
    model.add(Conv3D(128, kernel_size=(1,3,3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,2,2)))
    model.add(Conv3D(128, kernel_size=(1,2,2), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(10,activation='linear'))
    model.add(Dense(args.future_ghis+1,activation='linear'))
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

    model = get_3d_model_new(args)
    optimizer = optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    print(model.summary())
    # exit()
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # for epoch in range(args.epochs):
    # print("EPOCH ", epoch)
        
    sdl_train = SequenceDataLoaderMemChunks(args, catalog_train).batch(args.batch_size)
    sdl_val = SequenceDataLoaderMemChunks(args, catalog_val).batch(args.batch_size)
    # sdl_test = SequenceDataLoaderMemChunks(args, catalog_test)

    # history = model.fit_generator(generator=sdl_train,
    #                        validation_data=sdl_val,
    #                        workers=4)
    train_steps = args.train_steps//args.batch_size
    val_steps = args.val_steps//args.batch_size
    history = model.fit(x=sdl_train, validation_data=sdl_val, steps_per_epoch=train_steps, validation_steps=val_steps,
              epochs=args.epochs, workers=4)
    # val_result = model.evaluate(x=sdl_val)
    os.makedirs('history',exist_ok=True)
    print(history)
    with open('history/epoch_dict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # use_multiprocessing=True,
