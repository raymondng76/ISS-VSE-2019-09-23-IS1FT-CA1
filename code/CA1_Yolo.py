# ISS VSE CA1

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import pickle
import random as rand

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
from keras.engine.topology import Layer
from keras.utils import Sequence

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
    def fit(self, img_dir, annotation_dir):
        all_anno, labels = read_annotation_files(img_dir, annotation_dir)
    def fit_generator(self, img_dir, annotation_dir):
        all_anno, labels = read_annotation_files(img_dir, annotation_dir)        
    def predict(self):
        pass
#------------------------------------
#%%
#----------Generate Anchor boxes----------
def calculate_iou(anno, centroids):
    anno_width, anno_height = anno
    iou_list = []
    anno_area = anno_width * anno_height
    for cent in centroids:
        cent_width, cent_height = cent
        if cent_width <= anno_width and cent_height >= anno_height:
            iou = cent_width * anno_height / (anno_area + cent_width * (cent_height - anno_height))
        elif cent_width >= anno_width and cent_height >= anno_height:
            iou = anno_area / (cent_width * cent_height)
        elif cent_width >= anno_width and cent_height <= anno_height:
            iou = anno_width * cent_height / (anno_area + (cent_width - anno_width) * cent_height)
        else:
            iou = (cent_width * cent_height) / anno_area
        iou_list.append(iou)
    return np.array(iou_list)

def calculate_kMeans(anno_arr, num_anchor):
    curr_assign_centroids = np.zeros(anno_arr.shape[0])
    curr_distances = np.zeros((anno_arr.shape[0], num_anchor))
    indexes = [rand.randrange(anno_arr.shape[0]) for anchor in range(num_anchor)]
    sample_centroids = anno_arr[indexes]
    final = sample_centroids.copy()
    iterations = 0
    while True:
        iterations += 1
        distances = []
        for idx in range(anno_arr.shape[0]):
            distance = 1 - calculate_iou(anno_arr[idx], sample_centroids)
            distances.append(distance)
        distances_arr = np.array(distances)
        print(f'iter: {iterations} distances: {np.sum(np.abs(curr_distances-distances_arr))}')
        assign_centroids = np.argmin(distances_arr, axis=1)
        if (assign_centroids == curr_assign_centroids).all():
            return assign_centroids
        centroid_sums = np.zeros((num_anchor, anno_arr.shape[1]), np.float)
        for idx in range(anno_arr.shape[0]):
            centroid_sums[assign_centroids[idx]] += anno_arr[idx]
        for idx in range(num_anchor):
            sample_centroids[idx] = centroid_sums[idx] / (np.sum(assign_centroids==idx) + 1e-6)
        curr_assign_centroids = assign_centroids.copy()
        curr_distances = distances_arr.copy()

#Reference https://lars76.github.io/object-detection/k-means-anchor-boxes/
def generateAnchorBoxes(annotations, labels, num_anchor=9):
    annotation_dims = []
    for anno in annotations:
        for obj in anno['object']:
            rel_width = (float(obj['xmax']) - float(obj['xmin'])) / anno['width']
            rel_height = (float(obj['ymax']) - float(obj['ymin'])) / anno['height']
            annotation_dims.append(tuple(map(float,(rel_width, rel_height))))

    anno_arr = np.array(annotation_dims)
    centroids = calculate_kMeans(anno_arr, num_anchor)
    centroids_sorted = centroids[centroids[:,0].argsort()]
    anchor_box = []
    for cent in centroids_sorted:
        anchor_box.append(((int(cent[0]*416)),(int(cent[1]*416))))
    return anchor_box
#-----------------------------------------
#%%
#----------Data generator with Imgaug----------
#Refering to https://keras.io/utils/ (Sequence)
class DataGenerator(Sequence):
    '''Generate data with augmentations using Keras Util Sequence API
        Required Methods:
        def __init__(self)
        def __len__(self)
        def on_epoch_end(self)
        def __getitem__(self)'''

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
    
    # def generatesinglebatchdata(self, index):
    #     '''Generate a batch of data'''
    #     indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
    #     labels=np.array([self.labels[k] for k in indexes])
    #     images = [cv2.imread(self.image_path[k]) for k in indexes]
    
    #     if self.augment:
    #         images = self.augmentation(images)
    #     images = np.array([preprocess_input(img) for img in images])
    #     return images, labels
    
    def augmentation_with_boundingboxes(self, images, bbs):
        '''Image augmentation with imgaug'''
        seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=0.05*255),
            iaa.Affine(translate_px={"x": (1, 5)})
        ])
        return seq(images=images, bounding_boxes=bbs)
    
    def __getitem__(self, index):
        index = self.index[index * self.batch_size : (index + 1) * self.batch_size]
        labels = np.array
#----------------------------------------------
#%%
#----------VOC Parser----------
def read_annotation_files(image_dir, annnotation_dir):
    all_annotations = []
    labels = []
    for anno in sorted(os.listdir(annnotation_dir)):
        img = {'object':[]}
        try:
            xml_tree = ET.parse(annnotation_dir + anno)
        except Exception as ex:
            print(ex)
            print(f'Error parsing annotation xml: {annnotation_dir + anno}')
            continue
        for elem in xml_tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = image_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag:
                obj = {}
                for attribute in list(elem):
                    if 'name' in attribute.tag:
                        obj['name'] = attribute.text
                        if obj['name'] in labels:
                            labels[obj['name']] += 1
                        else:
                            labels[obj['name']] = 1
                        img['object'] += [obj]
                    if 'bndbox' in attribute.tag:
                        for dim in list(attribute):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
        if len(img['object']) > 0:
            bb_list = []
            for obj in img['object']:
                bb_list.append(ia.BoundingBox(x1=obj['xmin'], y1=obj['ymin'], x2=obj['xmax'], y2=obj['ymax'], label=obj['name']))
            img['bbs'] = ia.BoundingBoxesOnImage(bb_list, shape=(img['width'], img['height']))
            all_annotations += [img]
    return all_annotations, labels
#------------------------------
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
class YoloLossLayer(Layer):
    '''Use Keras API for layer creation to create custom layer for loss calculation
        Required methods:
        def __init__(self)
        def build(self)
        def call(self)'''
    def __init__(self, anchors, max_grid, batch_size, threshold, **kwargs):
        self.anchors = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        maxgrid_h, maxgrid_w = max_grid
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(maxgrid_w), [maxgrid_h]), (1, maxgrid_h, maxgrid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])
        self.threshold = threshold
        super(YoloLossLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(YoloLossLayer, self).build(input_shape)
    
    def call(self, x):
        input_img, y_pred, y_true, true_boxes = x

        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        object_mask     = tf.expand_dims(y_true[..., 4], 4)
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = tf.shape(input_img)[1]
        net_w       = tf.shape(input_img)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  
        pred_box_wh    = y_pred[..., 2:4]                                                       
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          
        pred_box_class = y_pred[..., 5:]                  
        true_box_xy    = y_true[..., 0:2]
        true_box_wh    = y_true[..., 2:4]
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)   
        conf_delta  = pred_box_conf - 0 
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious   = tf.reduce_max(iou_scores, axis=4)        
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        batch_seen = tf.assign_add(tf.Variable(0.), 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) + (1-object_mask) * conf_delta
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4)

        loss_xy    = tf.reduce_sum(tf.square(xy_delta), list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta), list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta), list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta, list(range(1,5)))
        return loss_xy + loss_wh + loss_conf + loss_class 
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

def YoloV3(numcls,anchors, max_grid, batch_size, threshold, max_boxes):
    img = Input(shape=(None,None,3))
    true_boxes = Input(shape=(1,1,1,max_boxes, 4))
    true_box_1 = Input(shape=(None, None, len(anchors)//6, 5+numcls))
    true_box_2 = Input(shape=(None, None, len(anchors)//6, 5+numcls))
    true_box_3 = Input(shape=(None, None, len(anchors)//6, 5+numcls))
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
        {'filters': (3*(5 + numcls)), 'kernel_size': 1, 'strides': 1, 'bnorm': True, 'leakyRelu': True}], skip=False)
    loss_small = YoloLossLayer(anchors=default_yolo_anchors[12:],
                                max_grid=[1*num for num in max_grid],
                                batch_size=batch_size,
                                threshold=threshold)([img, smallPred, true_box_1, true_boxes])
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
        {'filters': (3*(5 + numcls)), 'kernel_size': 1, 'strides': 1, 'bnorm': False, 'leakyRelu': False}], skip=False)
    loss_mid = YoloLossLayer(anchors=default_yolo_anchors[6:12],
                            max_grid=[2*num for num in max_grid],
                            batch_size=batch_size,
                            threshold=threshold)([img, midPred, true_box_2, true_boxes])
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
        {'filters': (3*(5+ numcls)), 'kernel_size': 1, 'strides': 1, 'bnorm': False, 'leakyRelu': False}], skip=False)
    loss_big = YoloLossLayer(anchors=default_yolo_anchors[:6],
                            max_grid=[4*num for num in max_grid],
                            batch_size=batch_size,
                            threshold=threshold)([img, bigPred, true_box_3, true_boxes])
    trainModel = Model([img, true_boxes, true_box_1, true_box_2, true_box_3], [loss_small, loss_mid, loss_big])
    inferModel = Model([img, smallPred, midPred, bigPred])
    return [trainModel, inferModel]
#---------------------------------

