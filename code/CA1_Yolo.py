# ISS VSE CA1

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

from keras import regularizers
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers.merge import add
from keras.layers.merge import concatenate
from keras.applications.nasnet import preprocess_input

#%%
#----------Config----------
NUM_CLASSES = 2
LABELS = ['car', 'bus']
#--------------------------
#%%
#---------- API for YoloV3 ----------
class YoloV3():
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs):
        pass
    def fit(self):
        pass
    def fit_generator(self):
        pass
    def predict(self):
        pass
#------------------------------------
#%%
#----------Data generator with Imgaug----------
class DataGenerator(keras.util.Sequence):
    '''Generate data with augmentations'''
    def __init__(self, image_path, labels, batch_size=32, image_dims=(416, 416, 3), shuffle=False, augment=True):
        self.image_path=image_path
        self.labels=labels
        self.batch_size=batch_size
        self.image_dims=image_dims
        self.shuffle=shuffle
        self.augment=augment
        self.on_epoch_end()
    
    def on_epoch_end(self):
        '''Update index after each epoch'''
        self.index = np.arange(len(self.image_path))
        if self.shuffle:
            np.random.shuffle(self.index)
    
    def generatesinglebatchdata(self, index):
        '''Generate a batch of data'''
        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        labels=np.array([self.labels[k] for k in indexes])
        images = [cv2.imread(self.image_path[k]) for k in indexes]
    
        if self.augment:
            images = self.augmentation(images)
        images = np.array([preprocess_input(img) for img in images])
        return images, labels
    
    def augmentation(self, images):
        '''Image augmentation with imgaug'''
        st = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), # Horizontal flip for 50% of all images
            iaa.Flipud(0.5), # Vertial flip for 50% of all images
            st(iaa.Affine(
                scale={"x":(0.9,1.1), "y":(0.9,1.1)}, # Scale images per axis
                translate_percent={"x":(-0.1,0.1), "y":(-0.1,0.1)}, # Translation per axis
                rotate=(-10,10), # Rotate by +/- 45 degrees
                shear=(-5,5), # Shear +/- 16 degrees
                order=[0,1],
                cval=[0,1],
                mode=ia.ALL
            )),
            iaa.SomeOf((0,5),
                        [st(iaa.Superpixels(p_replace=(0,1.0), n_segments=(20,200))),
                        iaa.OneOf([
                            iaa.GaussianBlur((0,1,0)),
                            iaa.AverageBlur(k=(3,5)),
                            iaa.MedianBlur(k=(3,5))
                        ]),
                        iaa.Sharpen(alpha=(0,1.0), lightness=(0.9,1.1)),
                        iaa.Emboss(alpha=(0,1.0), strength=(0,2.0)),
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5,1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5,1.0),
                                                    direction=(0.0,1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,0.1*255), per_channel=0.5),
                        iaa.OneOf([
                            iaa.Dropout((0.01,0.05), per_channel=0.5),
                            iaa.CoarseDropout((0.01,0.03), size_percent=(0.01,0.02), per_channel=0.2),
                        ]),
                        iaa.Invert(0.01, per_channel=True),
                        iaa.Add((-2,2), per_channel=0.5),
                        iaa.AddToHueAndSaturation((-1,1)),
                        iaa.OneOf([
                            iaa.Multiply((0.9,1.1), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(exponent=(-1,0), first=iaa.Multiply((0.9,1.1), per_channel=True), second=iaa.ContrastNormalization((0.9,1.1)))
                        ]),
                        st(iaa.ElasticTransformation(alpha=(0.5,3.5), sigma=0.25)),
                        st(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                        st(iaa.PerspectiveTransform(scale=(0.01,0.1)))
                        ], random_order=True),

        ], random_order=True)
        return seq.augment_images(images)
#----------------------------------------------
#%%
#---------- Class for Bounding Box + util functions ----------
class BoundingBox:
    '''Bounding Box definition'''
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax
        self.c=c
        self.classes=classes
        self.label = None
        self.score = None
    def get_label(self):
        if self.label == None:
            self.label = np.argmax(self.classes)
        return self.label
    def get_score(self):
        if self.score == None:
            self.score = self.classes[self.get_label()]
        return self.score

def getBoundBoxColor(label):
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255]
    ]
    if label < len(colors):
        return colors[label]
    else:
        return colors[0] # return default

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

def boundbox_iou(bx1, bx2):
    intsec_w = interval_overlap([bx1.xmin, bx2.xmax], [bx2.xmin, bx2.xmax])
    intsec_h = interval_overlap([bx1.ymin, bx2.ymax], [bx2.ymin, bx2.ymax])
    intsec = intsec_w * intsec_h
    w1, h1 = bx1.xmax-bx1.xmin, bx1.ymax-bx1.ymin
    w2, h2 = bx2.xmax-bx2.xmin, bx2.ymax-bx2.ymin
    union = w1*h1 + w2*h2 - intsec
    return float(intsec)/union

def draw_boundbox(img, boxes, labels, thresh):
    for box in boxes:
        label_str=''
        label = None
        for idx in range(len(labels)):
            if box.classes[idx] > thresh:
                if label_str != '': label_str +=', '
                label_str += (labels[idx] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = idx
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, img.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3, box.ymin],
                               [box.xmin-3, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin]], dtype='int32')
            cv2.rectangle(img=img, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=getBoundBoxColor(label), thickness=4)
            cv2.fillPoly(img=img, pts=[region], color=getBoundBoxColor(label))
            cv2.putText(img=img, text=label_str, org=(box.xmin+13, box.ymin-13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontscale=img.shape[0], color=(0, 0, 0), thickness=2)
    return img
#--------------------------------------------
#----------Loss Layer----------
# Refering to this https://keras.io/layers/writing-your-own-keras-layers/
# class YoloLossLayer():
    
#------------------------------
#%%
#---------- YOLOv3 Model----------
default_yolo_anchors = np.array([(10,13), (16,30), (33,23),
                                 (30,61), (62,45), (59,119),
                                 (116,90), (156,198), (373,326)],
                                 np.float32) / 416
default_yolo_anchors_mask = np.array([[6,7,8], [3,4,5], [0,1,2]])

def createYoloLyr(inputs, convLyrs, skip=True):
    '''Create yolo layer'''
    x = inputs
    loopCounter = 0
    for convLyr in convLyrs:
        if loopCounter == (len(convLyrs)-2) and skip:
            skip_connection = x
        loopCounter += 1
        if convLyr['strides'] > 1: 
            x = ZeroPadding2D(((1,0),(1,0)))(x)
        x = Conv2D(filters=convLyr['filters'], 
                   kernel_size=convLyr['kernel_size'], 
                   strides=convLyr['strides'], 
                   padding='valid' if convLyr['strides']> 1 else 'same',  
                   use_bias=False if convLyr['bnorm'] else True,
                   kernel_regularizer=regularizers.l2(0.0001))(x)
        if convLyr['bnorm']:
            x = BatchNormalization(epsilon=0.001)(x)
        if convLyr['leakyRelu']:
            x = LeakyReLU(alpha=0.1)(x)
    return add([skip_connection, x]) if skip else x

def YoloV3():
    img = Input(shape=(None,None,3))
    x = createYoloLyr(x, [
        {'filters': 32, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 64, 'kernel_size': 3, 'strides': 2, 'bnorm': True, 'leakyRelu': True},
        {'filters': 32, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 64, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    x = createYoloLyr(x, [
        {'filters': 128, 'kernel_size': 3, 'strides': 2, 'bnorm': True, 'leakyRelu': True},
        {'filters': 64, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 128, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    x = createYoloLyr(x, [
        {'filters': 64, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 128, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    x = createYoloLyr(x, [
        {'filters': 256, 'kernel_size': 3, 'strides': 2, 'bnorm': True, 'leakyRelu': True},
        {'filters': 128, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 256, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    for _ in range(7):
        x = createYoloLyr(x, [
            {'filters': 128, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
            {'filters': 256, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    out1 = x
    x = createYoloLyr(x, [
        {'filters': 512, 'kernel_size': 3, 'strides': 2, 'bnorm': True, 'leakyRelu': True},
        {'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 512, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    for _ in range(7):
        x = createYoloLyr(x, [
            {'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
            {'fitlers': 512, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    out2 = x
    x = createYoloLyr(x, [
        {'filters': 1024, 'kernel_size': 3, 'strides': 2, 'bnorm': True, 'leakyRelu': True},
        {'filters': 512, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 1024, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    for _ in range(3):
        x = createYoloLyr(x, [
            {'filters': 512, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
            {'filters': 1024, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
    x = createYoloLyr(x, [
        {'filters': 512, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 1024, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 512, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 1024, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 512, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True}], skip=False)
    smallPred = createYoloLyr(x, [
        {'filters': 1024, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': (3*(5 + NUM_CLASSES)), 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True}], skip=False)
    x = createYoloLyr(x, [{'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': False, 'leakyRelu': False}])
    x = UpSampling2D(2)(x)
    x = concatenate([x, out2])
    x = createYoloLyr(x, [
        {'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 512, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 512, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True}], skip=False)
    midPred = createYoloLyr(x, [
        {'filters': 512, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': (3*(5 + NUM_CLASSES)), 'kernel_size': 1, 'strides': 1, 'bnorm': False, 'leakyRelu': False}], skip=False)
    x = createYoloLyr(x, [
        {'filters': 128, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True}], skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, out1])
    bigPred = createYoloLyr(x, [
        {'filters': 128, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 256, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 128, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 256, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 128, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True},
        {'filters': (3*(5+ NUM_CLASSES)), 'kernel_size': 1, 'strides': 1, 'bnorm': False, 'leakyRelu': False}], skip=False)
    
#---------------------------------

