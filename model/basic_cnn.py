import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, ReLU
from data_loader import load_dataloader
from tqdm import tqdm
import pdb
import config
import wandb


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='None')
        self.batchnorm1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPool2D((2, 2))
        self.conv2 = Conv2D(64, 3, activation='None')
        self.batchnorm2 = BatchNormalization()
        self.conv3 = Conv2D(128, 3, activation='None')
        self.batchnorm3 = BatchNormalization()
        self.conv4 = Conv2D(256, 3, activation='None')
        self.batchnorm4 = BatchNormalization()
        self.conv5 = Conv2D(512, 5, activation='None')
        # self.batchnorm5 = BatchNormalization()
        self.flatten = Flatten()
        self.d1 = Dense(1, activation='linear')

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
        x = self.relu(x)
        # x = self.batchnorm5(x)
        x = self.flatten(x)
        x = self.d1(x)
        return x

# @tf.function


def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        # for i in range(10):
        # 	print(predictions[i].numpy(), labels[i].numpy())
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

    wandb.init(project="project1")

    args = config.init_args()

    print(tf.__version__)

    # Create an instance of the model
    model = MyModel()

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')

    for epoch in range(args.epochs):
        print("EPOCH ", epoch)
        train_loss.reset_states()
        valid_loss.reset_states()

        sdl_train, sdl_valid, sdl_test = load_dataloader(args)

        print(sdl_train)

        tm = tqdm(total=580)  # R! from data loader's tqdm
        ini = time.time()

        for images, labels in sdl_train:
            ini = time.time()
            train_step(images, labels)
            tm.update(1)
            ini = time.time()

        tm = tqdm(total=746 - 580)  # R! from data loader's tqdm
        for valid_images, valid_labels in sdl_valid:
            test_step(valid_images, valid_labels)
            tm.update(1)

        # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(
            "Epoch: ",
            epoch,
            "; Train Loss: ",
            train_loss.result(),
            "; Valid Loss: ",
            valid_loss.result())
        wandb.log({"Epoch": epoch,
                   "Train_Loss": train_loss.result(),
                   "Valid_Loss": valid_loss.result()})
        # print(template.format(epoch+1, train_loss.result(), 0))
