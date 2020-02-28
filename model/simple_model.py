import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Input, Dense, MaxPool2D, MaxPool3D, BatchNormalization, ReLU
from tensorflow.keras import Model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from utilities import config
from utilities.dataloader2 import SequenceDataLoaderMemChunks
from utilities.dataloader_simple import SimpleDataLoader, SimpleDataLoader2, SequenceDataLoader3, SimpleDataLoader4, SequenceDataLoader5, _generator
from utilities.utility import load_catalog
import pdb
from tqdm import tqdm


# Convolution 2D model for img(T0) -> GHI(T0) predictions
class MyModel(Model):
    """
    Args:
        args: argparse obj
    Returns:
        Built model
    """
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.input_0 = Input(shape=(70, 70, 5))
        self.conv1 = Conv2D(32, 3, input_shape=(70, 70, 5),
                            kernel_initializer='he_uniform')
        self.batchnorm1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPool2D((2, 2))
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
        # print(x,x[0])
        # x = self.input_0(x)
        x = self.conv1(x)  # 68 x 68 x 32
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 34 x 34 x 32
        x = self.conv2(x)  # 32 x 32 x 64
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 16 x 16 x 64
        x = self.conv3(x)  # 14 x 14 x 128
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 7 x 7 x 128
        x = self.conv4(x)  # 5 x 5 x 256
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.conv5(x)  # 1 x 1 x 512
        x = self.relu(x)
        x = self.batchnorm5(x)
        x = self.flatten(x)
        x = self.d1(x)
        return x  # Create an instance of the model

    def model(self):
        x = Input(shape=(70, 70, 5))
        return Model(inputs=[x], outputs=self.call(x))

# Convolution 2D model for img(T0) -> GHI(T0,...T0+6) predictions
class MyModel2(Model):
    """
    Args:
        args: argparse obj
    Returns:
        Built model
    """
    def __init__(self, args):
        super(MyModel2, self).__init__()
        self.input_0 = Input(shape=(70, 70, 5))
        self.conv1 = Conv2D(32, 3, input_shape=(70, 70, 5))
        self.batchnorm1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool2 = MaxPool2D((2, 2))
        self.maxpool3 = MaxPool2D((3, 3))
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.batchnorm2 = BatchNormalization()
        self.conv3 = Conv2D(128, 3, activation='relu')
        self.batchnorm3 = BatchNormalization()
        self.conv4 = Conv2D(256, 3, activation='relu')
        self.batchnorm4 = BatchNormalization()
        self.conv5 = Conv2D(512, 5, activation='relu')
        self.batchnorm5 = BatchNormalization()
        self.flatten = Flatten()
        self.d1 = Dense(10, activation='linear')
        self.d2 = Dense(4, activation='linear')

    def call(self, x):
        # x = x[0]
        # print(x,x[0])
        # x = self.input_0(x)
        x = self.conv1(x)  # 68 x 68 x 32
        x = self.batchnorm1(x)
        # x = self.relu(x)
        x = self.maxpool3(x)  # 34 x 34 x 32
        x = self.conv2(x)  # 32 x 32 x 64
        x = self.batchnorm2(x)
        # x = self.relu(x)
        x = self.maxpool2(x)  # 16 x 16 x 64
        x = self.conv3(x)  # 14 x 14 x 128
        x = self.batchnorm3(x)
        # x = self.relu(x)
        # x = self.maxpool2(x) # 7 x 7 x 128
        x = self.conv4(x)  # 5 x 5 x 256
        x = self.batchnorm4(x)
        # x = self.maxpool2(x) # 16 x 16 x 64
        # x = self.relu(x)
        x = self.conv5(x)  # 1 x 1 x 512
        # x = self.relu(x)
        x = self.batchnorm5(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x  # Create an instance of the model

    def model(self):
        x = Input(shape=(70, 70, 5))
        return Model(inputs=[x], outputs=self.call(x))

# CONV-3D

# Convolution 2D model for img(T0-n...T0) -> GHI(T0,...T0+6) predictions

class MyModel3(Model):
     """
    Args:
        args: argparse obj
    Returns:
        Built model
    """
    def __init__(self, args):
        super(MyModel3, self).__init__()
        self.conv1 = Conv3D(
            16, kernel_size=(
                1, 3, 3), input_shape=(
                5, 70, 70, 5))
        self.batchnorm1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPool3D((1, 2, 2))
        self.conv2 = Conv3D(32, kernel_size=(1, 3, 3))
        self.batchnorm2 = BatchNormalization()
        self.conv3 = Conv3D(64, kernel_size=(1, 3, 3))
        self.batchnorm3 = BatchNormalization()
        self.conv4 = Conv3D(128, kernel_size=(1, 3, 3))
        self.batchnorm4 = BatchNormalization()
        self.conv5 = Conv3D(256, kernel_size=(1, 5, 5))
        self.batchnorm5 = BatchNormalization()
        self.flatten = Flatten()
        self.d1 = Dense(4, activation='linear')

    def call(self, x):
        x = self.conv1(x)  # 68 x 68 x 32
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 34 x 34 x 32

        x = self.conv2(x)  # 32 x 32 x 64
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 16 x 16 x 64

        x = self.conv3(x)  # 14 x 14 x 128
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 7 x 7 x 128

        x = self.conv4(x)  # 5 x 5 x 256
        x = self.batchnorm4(x)
        x = self.relu(x)

        x = self.conv5(x)  # 1 x 1 x 512
        x = self.batchnorm5(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.d1(x)
        return x

    def model(self):
        # (seq, crop_h, crop_w, 5)
        x = Input(shape=(5, 70, 70, 5))
        return Model(inputs=[x], outputs=self.call(x))

# class for manual training of model using GradientTape instead of keras.fit
class new_model:

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            # for i in range(10):
            #   print(predictions[i].numpy(), labels[i].numpy())
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        print("\n Loss: ", loss.numpy())

    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        v_loss = self.loss_object(labels, predictions)

        self.valid_loss(v_loss)
        # print("\n Valid Loss: ", v_loss)

    def different_method(self):
        self.model = MyModel(args)
        self.model.build((None, 70, 70, 5))
        # pdb.set_trace()
        print(self.model.model().summary())
        # exit()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(args.lr)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        # catalog_train = load_catalog(args.data_catalog_path)
        # catalog_val = load_catalog(args.val_catalog_path)
        # catalog_test = load_catalog(args.test_catalog_path)

        for epoch in range(args.epochs):
            print("EPOCH ", epoch)
            self.train_loss.reset_states()
            self.valid_loss.reset_states()

            # .batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            sdl_train = SimpleDataLoader(args, args.data_catalog_path)
            # .batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            sdl_valid = SimpleDataLoader(args, args.val_catalog_path)

            train_total = 1000
            val_total = 100
            tm = tqdm(total=train_total)  # R! from data loader's tqdm
            # ini = time.time()

            counter = 0
            for batch in sdl_train:
                images = batch['images']
                labels = batch['y']
                self.train_step(images, labels,)
                tm.update(1)
                counter += 1
                if counter > train_total:
                    break
                # ini = time.time()

            counter = 0
            tm = tqdm(total=100)  # R! from data loader's tqdm
            for batch in sdl_valid:
                valid_images = batch['images']
                valid_labels = batch['y']
                self.test_step(valid_images, valid_labels)
                tm.update(1)
                counter += 1
                if counter > 100:
                    break
            # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(
                "Epoch: ",
                epoch,
                "; Train Loss: ",
                self.train_loss.result(),
                "; Valid Loss: ",
                self.valid_loss.result())
            # wandb.log({"Epoch":epoch,"Train_Loss":train_loss.result(),"Valid_Loss":valid_loss.result()})
            # print(template.format(epoch+1, train_loss.result(), 0))


def get_dataloaders(datal, args):
    # .prefetch(tf.data.experimental.AUTOTUNE).batch(args.batch_size)
    sdl_train = datal(args, args.data_catalog_path)
    # .prefetch(tf.data.experimental.AUTOTUNE).batch(args.batch_size)
    sdl_val = datal(args, args.val_catalog_path)
    return sdl_train, sdl_val


def train_simple_model(args):

    model = MyModel2(args)

    optimizer = Adam(learning_rate=args.lr)
    rmse = RootMeanSquaredError()
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[rmse])
    model.build((None, 70, 70, 5))
    # pdb.set_trace()
    print(model.model().summary())
    # return model, SimpleDataLoader4
    return model, SequenceDataLoader5


def train_seq_model(args):

    model = MyModel3(args)

    optimizer = Adam(learning_rate=args.lr)
    rmse = RootMeanSquaredError()
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[rmse])
    model.build((None, 1, 70, 70, 5))
    # pdb.set_trace()
    print(model.model().summary())
    return model, SequenceDataLoader3


if __name__ == "__main__":
    args = config.init_args()

    # nm = new_model()
    # nm.different_method()
    # exit()

    model, dataloader = train_simple_model(args)
    # model, dataloader = train_seq_model(args)

    sdl_train, sdl_val = get_dataloaders(dataloader, args)

    model.fit(
        sdl_train,
        # steps_per_epoch=100,
        epochs=args.epochs,
        validation_data=sdl_val,
        # validation_steps=100,
    )
    # model.save('very_very_confidential_model',save_format='tf')
