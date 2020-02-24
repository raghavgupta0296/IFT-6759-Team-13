import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, MaxPool3D, BatchNormalization, ReLU
from tqdm import tqdm
import pdb
from utilities import config
from utilities.dataloader import SequenceDataLoaderMemChunks
from utilities.utility import load_catalog
import numpy as np
import wandb

wandb.init(project="project1-seq")

args = config.init_args()

print(tf.__version__)

def preprocess(x,y):
    if args.normalize_img:
        x = tf.image.per_image_standardization(x)
    if args.normalize_y:
        y = tf.math.divide(y,10)
    return x,y

class MyModel(Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = Conv3D(32, kernel_size=(1,3,3), input_shape=(5,70,70,5), kernel_initializer='he_uniform')
		self.batchnorm1 = BatchNormalization()
		self.relu = ReLU()
		self.maxpool = MaxPool3D((1,2,2))
		self.conv2 = Conv3D(64, kernel_size=(1,3,3), kernel_initializer='he_uniform')
		self.batchnorm2 = BatchNormalization()
		self.conv3 = Conv3D(128, kernel_size=(1,3,3), kernel_initializer='he_uniform')
		self.batchnorm3 = BatchNormalization()
		self.conv4 = Conv3D(256, kernel_size=(1,3,3), kernel_initializer='he_uniform')
		self.batchnorm4 = BatchNormalization()
		self.conv5 = Conv3D(512, kernel_size=(1,5,5), kernel_initializer='he_uniform')
		self.batchnorm5 = BatchNormalization()
		self.flatten = Flatten()
		self.d1 = Dense(4, activation='linear')

	def call(self, x):
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
		return x


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')

# tform_data = np.load("transformation_stats.npz")
# y_mov_avg = tform_data['y_mov_avg']
# y_mov_std = tform_data['y_mov_std']

# @tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images, training=True)
		loss = loss_object(labels, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	train_loss(loss_object(labels, predictions))
	print("\n Loss: ", loss.numpy())
	for i in range(10):
		print(predictions[i].numpy(), labels[i].numpy())

# @tf.function
def test_step(images, labels):
	predictions = model(images, training=False)
	v_loss = loss_object(labels, predictions)
	valid_loss(loss_object(labels, predictions))
	print("\n ** Valid Loss: ", v_loss.numpy())
	for i in range(10):
		print(predictions[i].numpy(), labels[i].numpy())

wandb.config.learning_rate = optimizer.get_config()['learning_rate']
	
for epoch in range(args.epochs):
	print("EPOCH ", epoch)
	train_loss.reset_states()
	valid_loss.reset_states()

	catalog_train = load_catalog(args.data_catalog_path)
	catalog_val = load_catalog(args.val_catalog_path)
	catalog_test = load_catalog(args.test_catalog_path)

	sdl_train = SequenceDataLoaderMemChunks(args, catalog_train).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE).batch(64)
	sdl_val = SequenceDataLoaderMemChunks(args, catalog_val).map(preprocess).prefetch(tf.data.experimental.AUTOTUNE).batch(64)
	
	print(sdl_train)

	# tm = tqdm(total=580)  # R! from data loader's tqdm
	# ini = time.time()

	for images, labels in sdl_train:
		# print("~~~~~~~~~~~~~~~~~GOT from dataloader in ", time.time()-ini)
		# print("training...")
		# ini = time.time()
		train_step(images, labels)
		print("Train Loss: ", train_loss.result())
		# print("one iteration done in ", time.time() - ini, "... loading data again")
		# tm.update(1)
		# ini = time.time()
		print(images.shape,labels.shape)

	# tm = tqdm(total=746-580)  # R! from data loader's tqdm
	for valid_images, valid_labels in sdl_valid:
		test_step(valid_images, valid_labels)
		print(images.shape,labels.shape)
		# tm.update(1)

	template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
	print("Epoch: ", epoch, "; Train Loss: ", train_loss.result(), "; Valid Loss: ", valid_loss.result())
	wandb.log({"Epoch":epoch,"Train_Loss":train_loss.result(),"Valid_Loss":valid_loss.result()})

model.summary()
