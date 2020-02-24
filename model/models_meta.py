import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, ReLU, Embedding, Concatenate, GlobalMaxPooling2D
from data_loader_meta import load_dataloader
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
import pdb
import config
import numpy as np
import wandb

class Resnet_Meta(Model):
	def __init__(self,args):
		super(MyModel, self).__init__()
		self.resnet_base = ResNet50(include_top=False, weights=None, input_shape=(70, 70, 5), pooling='avg')
		self.d1 = Dense(64, activation='relu')

		self.embedding1 = Embedding(12,4) # month
		self.embedding2 = Embedding(31,4) # weekday
		self.embedding3 = Embedding(24,5) # time
		self.d2 = Dense(64, activation='relu')
		
		self.concatenation = Concatenate()
		self.concatenation2 = Concatenate()

		self.d3 = Dense(64, activation='relu')
		self.d4 = Dense(args.future_ghis+1, activation='linear')


	def call(self, x):
		x2 = x
		x = x[0]
		
		# x = preprocess_input(x)
		x = self.resnet_base(x)
		# x = self.global_pooling(x)
		x = self.d1(x)

		x2_1 = self.embedding1(x2[1])
		x2_2 = self.embedding2(x2[2])
		x2_3 = self.embedding3(x2[3])

		x2 = self.concatenation([x2_1, x2_2, x2_3, x2[4],x2[5],x2[6],x2[7]])
		x2 = self.d2(x2)

		x = self.concatenation2([x,x2])
		
		x = self.d3(x)
		x = self.d4(x)

		return x

class Basic_CNN_Meta(Model):
	def __init__(self,args):
		super(MyModel, self).__init__()
		self.conv1 = Conv2D(32, 3)
		self.batchnorm1 = BatchNormalization()
		self.relu = ReLU()
		self.maxpool = MaxPool2D((2,2))
		self.conv2 = Conv2D(64, 3)
		self.batchnorm2 = BatchNormalization()
		self.conv3 = Conv2D(128, 3)
		self.batchnorm3 = BatchNormalization()
		self.conv4 = Conv2D(256, 3)
		self.batchnorm4 = BatchNormalization()
		self.conv5 = Conv2D(512, 5)
		self.batchnorm5 = BatchNormalization()
		self.flatten = Flatten()
		self.d1 = Dense(64, activation='linear')

		self.embedding1 = Embedding(12,4) # month
		self.embedding2 = Embedding(31,4) # weekday
		self.embedding3 = Embedding(24,5) # time
		
		self.concatenation = Concatenate()

		self.d2 = Dense(args.future_ghis+1, activation='linear')


	def call(self, x):
		x2 = x
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

		x2_1 = self.embedding1(x2[1])
		x2_2 = self.embedding2(x2[2])
		x2_3 = self.embedding3(x2[3])
		
		x = self.concatenation([x, x2_1, x2_2, x2_3,x2[4],x2[5],x2[6],x2[7]])

		x = self.d2(x)

		return x
