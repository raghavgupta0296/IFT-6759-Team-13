import tensorflow as tf
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


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np




NB_OF_NEURONS = 100

def create_resnet50_model(input_shape, top='flatten'):
    if top not in ('flatten', 'avg', 'max'):
        raise ValueError('unexpected top layer type: %s' % top)
    # connects base model with new "head"
    BottleneckLayer = {
        'flatten': Flatten(),
        'avg': GlobalAvgPooling2D(),
        'max': GlobalMaxPooling2D()
    }[top]
    
    base = ResNet50(include_top=False, input_shape=input_shape)
    x = BottleneckLayer(base.output)
    x = Dense(NB_OF_NEURONS, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=x)
    return model

