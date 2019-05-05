from keras.engine.saving import load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

class SRGAN():
    def __init__(self):
        self.channels = 3
        self.lr_height = 192
        self.lr_width = 384
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height
        self.hr_width = self.lr_width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        optimizer = Adam(0.0001)
        """
        pre-trained VGG19 model to extract image features from the high resolution image 
        and the generated high resolution images and minimize the mse between them
        """
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch * 2, 1)

        # Number of filters in the first layer of Gen and Disc
        self.gf = 64
        self.df = 32

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        fake_hr = self.generator(img_lr)

        fake_features = self.vgg(fake_hr)

        self.discriminator.trainable = False

        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def build_vgg(self):

        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.5)(d)
            d = Activation('relu')(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.5)(d)
            d = Add()([d, layer_input])
            return d

        img_lr = Input(shape=self.lr_shape)

        c1 = Conv2D(self.gf, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        c2 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.5)(c2)
        c2 = Add()([c2, c1])

        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(c2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.5)(d)
            return d

        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d1 = d_block(d1, self.df, strides=2)
        d1 = d_block(d1, self.df * 2)
        d1 = d_block(d1, self.df * 2, strides=2)
        d1 = d_block(d1, self.df * 4)
        d1 = d_block(d1, self.df * 4, strides=2)
        d1 = d_block(d1, self.df * 8)
        d1 = d_block(d1, self.df * 8, strides=2)

        d1 = Dense(self.df * 8)(d1)

        d1 = LeakyReLU(alpha=0.2)(d1)
        validity = Dense(1, activation='sigmoid')(d1)

        return Model(d0, validity)

    def pred(self, count=0, idx=0):
        test = cv2.imread("/content/test.jpg")
        test = load_image_test(test)
        result = self.generator.predict(test)
        result = (result + 1) * 127.5
        result = np.array(result, dtype=np.uint8)
        cv2.imwrite("testout%d.jpg" % (count), result[0])

        result = self.generator.predict([[lr[idx]]])
        result = (result + 1) * 127.5
        result = np.array(result, dtype=np.uint8)
        cv2.imwrite("output%d.jpg" % (count), result[0])

        result = (lr[idx] + 1) * 127.5
        result = np.array(result, dtype=np.uint8)
        cv2.imwrite("input%d.jpg" % (count), result)

    def train(self, gw, dw, gepochs, depochs, batch_size=4, saved=False):

        if saved == True:
            self.generator.load_weights(gw)
            self.discriminator.load_weights(dw)

        #  Train Discriminator
        fake_hr = self.generator.predict(lr)

        valid = np.ones((np.shape(hr)[0],) + self.disc_patch)
        fake = np.zeros((np.shape(hr)[0],) + self.disc_patch)

        print("Train the discriminators")
        # Train the discriminators (original images = real / generated = Fake)
        self.discriminator.fit(hr, valid, verbose=1, batch_size=batch_size, epochs=depochs)
        self.discriminator.fit(fake_hr, fake, verbose=1, batch_size=batch_size, epochs=depochs * 2)

        #  Train Generator

        # Extract features from hr image using pre-trained VGG19 model
        image_features = self.vgg.predict(hr)

        print("Train the generators")
        g_loss = self.combined.fit([lr, hr], [valid, image_features], verbose=1, batch_size=batch_size, epochs=gepochs)
        self.generator.save_weights("gdrive/My Drive/Colab Notebooks/generator.h5")
        self.discriminator.save_weights("gdrive/My Drive/Colab Notebooks/discriminator.h5")

dl = DataLoader("license_plates")
hr, lr = dl.load_data()

dl = DataLoader("IRCP_dataset_1024X768")
hr2, lr2 = dl.load_data()

hr = np.concatenate((hr, hr2))
lr = np.concatenate((lr, lr2))

gan = SRGAN()
gan.train(_,_,gepochs=50,depochs=10,batch_size=4)