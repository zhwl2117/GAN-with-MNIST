from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import os


class DCGAN:
    def __init__(self):
        self.row = 28
        self.col = 28
        self.channel = 1
        self.img_shape = (self.row, self.col, self.channel)
        self.latent_size = 100

        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        z = Input(shape=(100,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation='relu', input_dim=self.latent_size))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())  # check up sample
        model.add(Conv2D(128, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Conv2D(self.channel, kernel_size=3, padding='same'))
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.latent_size,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape,
                         padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))  # check this function
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        (X_train, _), (_, _) = mnist.load_data()
        # X_train = X_train / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)  # check this function
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            images = X_train[idx] / 127.5 - 1
            noises = np.random.uniform(-1, 1, (batch_size, self.latent_size))
            gen_image = self.generator.predict(noises)

            # eval_acc_real = self.discriminator.test_on_batch(images, valid)[1]
            # eval_acc_fake = self.discriminator.test_on_batch(gen_image, fake)[1]
            # eval_acc = 0.5 * (eval_acc_fake + eval_acc_real)
            d_loss_real = self.discriminator.train_on_batch(images, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_image, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = self.combined.train_on_batch(noises, valid)
            g_eval_acc = self.discriminator.test_on_batch(gen_image, valid)[1]
            if d_loss_real[1] > 0.9 and d_loss_fake[1] > 0.7 and g_eval_acc > 0.9:
                self.generator.save('./GAN/generator_model_optimal.h5')
            if epoch % 10 == 0:
                print('{}:{}, {}\t{}'.format(epoch, d_loss_real[1], d_loss_fake[1], g_eval_acc))

    def predict(self):
        return self.generator.predict(np.random.normal(0, 1, (128, self.latent_size)))

    def save(self):
        # os.mkdir('./GAN')
        self.generator.save('./GAN/generator_model_fixed1.h5')
        self.combined.save('./GAN/GAN_combined_model_fixed1.h5')


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=3000)
    dcgan.save()
