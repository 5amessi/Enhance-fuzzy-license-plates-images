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
from data_loader import DataLoader
def Cnn():
    n_residual_blocks = 16

    def residual_block(layer_input, filters):
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = BatchNormalization(momentum=0.5)(d)
        d = Activation('relu')(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.5)(d)
        d = Add()([d, layer_input])
        return d

    img_lr = Input(shape=(240,480,3))

    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
    c1 = Activation('relu')(c1)

    r = residual_block(c1, 64)
    for _ in range(n_residual_blocks - 1):
        r = residual_block(r, 64)

    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
    c2 = BatchNormalization(momentum=0.5)(c2)
    c2 = Add()([c2, c1])

    gen_hr = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(c2)

    return Model(img_lr, gen_hr)

dl = DataLoader("license_plates")
hr, lr = dl.load_data()

dl = DataLoader("IRCP_dataset_1024X768")
hr2, lr2 = dl.load_data()

hr = np.concatenate((hr, hr2))
lr = np.concatenate((lr, lr2))

model = Cnn()
optimizer = keras.optimizers.Adam(0.0001)
model.compile(loss='mse',
            optimizer=optimizer)
model.fit(x=lr,y=hr,batch_size=4,epochs=100,verbose=1)

