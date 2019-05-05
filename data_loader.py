import scipy
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *


class DataLoader():
    def __init__(self, dataset_name, img_res=(480, 240), out_res=(480, 240)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.out_res = out_res

    def load_data(self, png=False):
        if png == True:
            path = glob('/content/%s/*.png' % (self.dataset_name))
        else:
            path = glob('/content/%s/*' % (self.dataset_name))

        imgs_hr = []
        imgs_lr = []

        for idx, i in enumerate(path):
            if idx >= 1000:
                break
            img = cv2.imread(i)
            w, h = self.img_res
            low_w, low_h = int(w / 4), int(h / 4)

            img_hr = cv2.resize(img, self.out_res)

            img_lr = cv2.resize(img, (low_w, low_h))
            img_lr = cv2.resize(img_lr, self.img_res)

            flr = np.fliplr(img_lr)
            fhr = np.fliplr(img_hr)

            imgs_hr.append(img_hr)
            imgs_hr.append(fhr)
            imgs_lr.append(img_lr)
            imgs_lr.append(flr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr
