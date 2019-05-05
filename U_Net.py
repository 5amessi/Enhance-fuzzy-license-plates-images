import scipy
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *

def conv_block(prevlayer, filters, kernel, prefix, strides=(1,1)):
	conv = Conv2D(filters, kernel, padding="same",  strides=strides,
				  name=prefix + "_conv")(prevlayer)
	conv = BatchNormalization(name=prefix + "_bn")(conv)
	conv = Activation('relu', name=prefix + "_activation")(conv)
	#conv = LeakyReLU(0.2, name=prefix + "_activation")(conv)
	return conv


def UNET():

    img_input = Input(shape=(240,480,3))

    conv1 = conv_block(img_input, 128, (3,3), "conv1_1")
    pool1 = MaxPooling2D(2, strides=(2,2), padding="same", name="pool1")(conv1)

    conv2 = conv_block(pool1, 256, (3,3), "conv2_1")
    pool2 = MaxPooling2D(2, strides=(2,2), padding="same", name="pool2")(conv2)

    conv3 = conv_block(pool2, 512, (3,3), "conv3_1")
    pool3 = MaxPooling2D(2, strides=(2,2), padding="same", name="pool3")(conv3)

    #####################################################
    conv4 = conv_block_simple_down(pool3, 512, (3,3), "conv4_1")
    #####################################################

    up5 = concatenate([UpSampling2D()(conv4), conv3])
    conv5 = conv_block(up5, 512, (3,3), "conv5_1")

    up6 = concatenate([UpSampling2D()(conv5), conv2])
    conv6 = conv_block(up6, 256, (3,3), "conv6_1")

    up7 = concatenate([UpSampling2D()(conv6), conv1])
    conv7 = conv_block(up7, 128, (3,3), "conv7_1")

    conv8 = conv_block(conv7, 64, (3,3), "conv7_2")

    prediction = Conv2D(3, (3,3), activation="tanh", padding="same", name="prediction", kernel_initializer="Orthogonal")(conv8)

    model = Model(img_input, prediction)
    return model

dl = DataLoader("license_plates")
hr, lr = dl.load_data()

dl = DataLoader("IRCP_dataset_1024X768")
hr2, lr2 = dl.load_data()

hr = np.concatenate((hr, hr2))
lr = np.concatenate((lr, lr2))

model = UNET()
optimizer = keras.optimizers.Adam(0.0001)
model.compile(loss='mse',
            optimizer=optimizer)
model.fit(x=lr,y=hr,batch_size=8,epochs=20,verbose=1)