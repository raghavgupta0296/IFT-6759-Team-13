import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import wandb
import pickle
import os
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers, Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, BatchNormalization
from tensorflow.keras import optimizers

from utilities.dataloader import SequenceDataLoader, SequenceDataLoaderMemChunks
from utilities import config


def get_3d_model():
    model = Sequential()
    model.add(
        Conv3D(
            16, kernel_size=(
                2, 3, 3), activation='relu', input_shape=(
                2, 70, 70, 5)))
    # model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 3, 3)))
    model.add(Conv3D(16, kernel_size=(1, 5, 5), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 3, 3)))
    model.add(Conv3D(10, kernel_size=(1, 5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='linear'))
    model.add(Dense(2, activation='linear'))
    return model


def get_3d_model_new(args):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(args.k_sequences + 1, 5, 5),
                     padding='same', activation='relu', input_shape=(args.k_sequences + 1, 70, 70, 5)))
    model.add(MaxPooling3D(pool_size=(1, 3, 3)))
    model.add(
        Conv3D(
            128,
            kernel_size=(
                1,
                3,
                3),
            padding='same',
            activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 3, 3)))
    model.add(
        Conv3D(
            256,
            kernel_size=(
                1,
                3,
                3),
            padding='same',
            activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(
        Conv3D(
            512,
            kernel_size=(
                1,
                2,
                2),
            padding='same',
            activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='linear'))
    model.add(Dense(args.future_ghis + 1, activation='linear'))
    return model


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
    optimizer = optimizers.Adam(
        learning_rate=args.lr,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=False)
    print(model.summary())
    # exit()
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # for epoch in range(args.epochs):
    # print("EPOCH ", epoch)
    def preprocess(x, y):
        if args.normalize_img:
            x = tf.image.per_image_standardization(x)
        if args.normalize_y:
            y = tf.math.divide(y, 10)
        return x, y
    sdl_train = SequenceDataLoaderMemChunks(
        args, catalog_train).map(preprocess).batch(
        args.batch_size)
    sdl_val = SequenceDataLoaderMemChunks(
        args, catalog_val).map(preprocess).batch(
        args.batch_size)
    # sdl_test = SequenceDataLoaderMemChunks(args, catalog_test)

    train_steps = args.train_steps // args.batch_size
    val_steps = args.val_steps // args.batch_size
    history = model.fit(x=sdl_train, validation_data=sdl_val, steps_per_epoch=train_steps, validation_steps=val_steps,
                        epochs=args.epochs)
    # val_result = model.evaluate(x=sdl_val)
    os.makedirs('history', exist_ok=True)
    print(history)
    with open('history/epoch_dict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    # use_multiprocessing=True,
