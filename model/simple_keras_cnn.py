import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, ReLU
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from dataloader_simple import SimpleDataLoader2
import pickle
import time

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

optimizer = Adam(learning_rate=0.01)
rmse = RootMeanSquaredError()

model.compile(optimizer=optimizer, 
			  loss='mean_squared_error',
			  metrics=[rmse])

chkpt = tf.keras.callbacks.ModelCheckpoint(
"keras_models/keras_model_1", monitor='val_loss', verbose=0, save_best_only=True,
save_weights_only=False, mode='auto', save_freq='epoch'
)
tfboard = tf.keras.callbacks.TensorBoard(
log_dir="keras_models/logs_keras_model", histogram_freq=0, write_graph=True, write_images=False,
update_freq='epoch', profile_batch=2, embeddings_freq=0,
embeddings_metadata=None)

sdl_train = SimpleDataLoader2("train")
sdl_valid = SimpleDataLoader2("valid")

model.fit_generator(
	sdl_train, 
	# steps_per_epoch=None,
	epochs=10, 
	validation_data=sdl_valid,
	# validation_steps=None,
    callbacks=[chkpt,tfboard]
	)
