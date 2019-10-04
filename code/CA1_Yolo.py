# ISS VSE CA1


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers.merge import add, concatenate

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
        for idx in range(len(lables)):
            if box.classes[i] > thresh:
                if label_str != '': label_str +=', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3, box.ymin],
                               [box.xmin-3, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin]], dtype='int32')
            cv2.rectangle(img=img, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=getBoundBoxColor(label), thickness=4)
            cv2.fillPoly(img=img, pts=[region], color=getBoundBoxColor(label))
            cv2.putText(img=img, text=label_str, org=(box.xmin+13, box.ymin-13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontscale=image.shape[0], color=(0, 0, 0), thickness=2)
    return img
#--------------------------------------------
#---------- YOLOv3 Model----------
def CreateYoloLyr(inputs, convLyrs, skip=True):
    '''Create yolo layer'''
    x = inputs
    loopCounter = 0
    for convLyr in convLyrs:
        if loopCounter == (len(convLyrs)-2) and skip:
            skip_connection = x
        loopCounter += 1
        if convLyr['strides'] > 1: 
            x = ZeroPadding2D(((1,0),(1,0)), name='zeropad_'+str(convLyr['lyrName']))(x)
        x = Conv2D(filters=convLyr['filters'], 
                   kernel_size=convLyr['kernel_size'], 
                   strides=convLyr['strides'], 
                   padding='valid' if convLyr['strides']> 1 else 'same',  
                   use_bias=False if convLyr['bnorm'] else True, 
                   name='conc2d_'+str(convLyr['lyrName']))(x)
        if convLyr['bnorm']:
            x = BatchNormalization(epsilon=0.001, name='bnorm_idx_'+str(convLyr['lyrName']))(x)
        if convLyr['leakyRelu']:
            x = LeakyReLU(alpha=0.1, name='leakyrelu_idx_'+str(convLyr['lyrName']))(x)
    return add([skip_connection, x]) if skip else x
#---------------------------------