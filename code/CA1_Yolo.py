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
import copy

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
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import BaseLogger
from keras.callbacks import ModelCheckpoint

#%%
#---------- API for YoloV3 ----------
class YoloV3():
    def __init__(self, *args, **kwargs):
        '''Ctor'''
        # super().__init__(*args, **kwargs):
        pass
    def fit(self, img_dir, annotation_dir, train_size):
        '''Fit without augmentation'''
        all_anno, labels, anchor_boxes = self._process_dataset(img_dir, annotation_dir)
        train_anno, valid_anno, max_boxes = _train_valid_split(all_anno)
    def fit_generator(self, img_dir, annotation_dir, train_size):
        '''Fit with augmentation'''
        all_anno, labels, anchor_boxes = self._process_dataset(img_dir, annotation_dir)
        train_anno, valid_anno, max_boxes = self._train_valid_split(all_anno)
        train_generator = DataGenerator(
            annotations=train_anno,
            max_boxes=max_boxes,
            anchors=anchor_boxes,
            labels=labels,
            batch_size=16,
            width=416,
            height=416,
            shuffle=True,
            augment=True)
        valid_generator = DataGenerator(
            annotations=valid_anno,
            max_boxes=max_boxes,
            anchors=anchor_boxes,
            labels=labels,
            batch_size=16,
            width=416,
            height=416,
            shuffle=True,
            augment=True)
        
        train_model, infer_mode = YoloV3(
            numcls=len(labels),
            anchors=anchor_boxes,
            max_grid=[416, 416],
            batch_size=16,
            threshold=0.5,
            max_boxes=max_boxes)
        callback = create_callbacks()
        train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator) * 3,
            epochs=300,
            callbacks=callback,
            workers=4,
            max_queue_size=8)
        
    def predict(self):
        pass
    def _process_dataset(self, img_dir, annotation_dir):
        all_anno, labels = read_annotation_files(img_dir, annotation_dir)
        anchor_boxes = generateAnchorBoxes(all_anno, labels)
        return all_anno, labels, anchor_boxes
    def _train_valid_split(self, all_anno, train_size=0.8):
        train_valid_split = int(train_size * len(all_anno))
        np.random.shuffle(all_anno)
        train_anno = all_anno[:train_valid_split]
        valid_anno = all_anno[train_valid_split:]
        max_boxes = max([len(anno['object']) for anno in (train_anno + valid_anno)])
        return train_anno, valid_anno, max_boxes
#------------------------------------
#%%
#----------VOC Parser----------
def read_annotation_files(image_dir, annnotation_dir):
    all_annotations = []
    labels = {}
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
    return all_annotations, labels.keys()
#------------------------------
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
            return sample_centroids
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
        anchor_box.append(int(cent[0]*416))
        anchor_box.append(int(cent[1]*416))
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

    def __init__(self, annotations, max_boxes, anchors, labels, batch_size=32, width=416, height=416, shuffle=False, augment=True):
        self.annotations=annotations #instance
        self.max_boxes=max_boxes
        self.anchors=[ia.BoundingBox(x1=0.0,y1=0.0,x2=anchors[2*i],y2=anchors[2*i+1]) for i in range(len(anchors)//2)]
        self.labels=labels
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.augment=augment
        self.basefactor=32
        self.width=width
        self.height=height
        self.min_size=320
        self.max_size=608
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(float(len(self.annotations))//self.batch_size))

    def on_epoch_end(self):
        '''Update index after each epoch'''
        self.index = np.arange(len(self.annotations))
        if self.shuffle:
            np.random.shuffle(self.index)
    
    def augmentation_with_boundingboxes(self, images, bbs):
        '''Image augmentation with imgaug'''
        seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=0.05*255),
            iaa.Affine(translate_px={"x": (1, 5)})
        ])
        return seq(images=images, bounding_boxes=bbs)
    
    def _current_size(self, idx):
        if idx%10 == 0:
            net_size = self.basefactor*np.random.randint(self.min_size/self.basefactor, \
                                                         self.max_size/self.basefactor+1)
            self.height, self.width = net_size, net_size
        return self.height, self.width

    def _limit_constraint(self, value, minVal, maxVal):
        if value < minVal:
            return minVal
        if value > maxVal:
            return maxVal
        return value

    def _scale_boxes(self, boxes, img_height, img_width, new_height, new_width):
        all_boxes = copy.deepcopy(boxes)
        xScale, yScale = float(new_width) / img_width, float(new_height) / img_height
        box_list = []
        for box in boxes.bounding_boxes:
            # scale_box = ia.BoundingBox(
            #     x1=int(self._limit_constraint(0, new_width, box.x1 * sx)),
            #     x2=int(self._limit_constraint(0, new_width, box.x2 * sx)),
            #     y1=int(self._limit_constraint(0, new_height, box.y1 * sy)),
            #     y2=int(self._limit_constraint(0, new_height, box.y2 * sy)),
            #     label=box.label)
            scale_box = ia.BoundingBox(
                x1=int(np.round(box.x1 * xScale)),
                x2=int(np.round(box.x2 * xScale)),
                y1=int(np.round(box.y1 * yScale)),
                y2=int(np.round(box.y2 * yScale)),
                label=box.label)
            # if (scale_box.x2 <= scale_box.x1 or scale_box.y2 <= scale_box.y1):
            #     continue
            box_list.append(scale_box)
        return ia.BoundingBoxesOnImage(box_list, (new_width, new_height))
    
    def __getitem__(self, index):
        '''Get input per batch'''
        self.height, self.width = self._current_size(index)
        grid_height, grid_width = self.height//self.basefactor, self.width//self.basefactor
        curr_indices = index * self.batch_size # r_bound
        next_indices = (index + 1) * self.batch_size # l_bound
        if curr_indices > len(self.annotations):
            curr_indices = len(self.annotations)
            next_indices = curr_indices - self.batch_size
        input_images = np.zeros((next_indices - curr_indices, self.height, self.width, 3))
        groundtruths = np.zeros((next_indices - curr_indices, 1, 1, 1, self.max_boxes, 4))

        yolo_smallout = np.zeros((next_indices - curr_indices, grid_height, grid_width, len(self.anchors)//3, 5+len(self.labels)))
        yolo_midout = np.zeros((next_indices - curr_indices, 2 * grid_height, 2 * grid_width, len(self.anchors)//3, 5+len(self.labels)))
        yolo_bigout = np.zeros((next_indices - curr_indices, 4 * grid_height, 4 * grid_width, len(self.anchors)//3, 5+len(self.labels)))
        all_out = [yolo_bigout, yolo_midout, yolo_smallout]
        yolo_loss1 = np.zeros((next_indices - curr_indices, 1))
        yolo_loss2 = np.zeros((next_indices - curr_indices, 1))
        yolo_loss3 = np.zeros((next_indices - curr_indices, 1))

        true_box_idx = 0
        img_count = 0
        for anno in self.annotations[curr_indices:next_indices]: #Each image and annotations for current batch
            raw_img = cv2.imread(anno['filename'])
            raw_img = cv2.resize(raw_img, (self.width, self.height))
            boundboxes = anno['bbs']
            boundboxes = self._scale_boxes(boundboxes, raw_img.shape[0], raw_img.shape[1], self.height, self.width)
            img, bbs = self.augmentation_with_boundingboxes(raw_img, boundboxes)
            for box in bbs.bounding_boxes:
                max_anchor, max_index = self._get_best_anchor(box)
                yolo_out = all_out[max_index//3]
                yolo_grid_height, yolo_grid_width = yolo_out.shape[1:3]
                centerX = (0.5 * (box.x1 + box.x2)) / (float(self.width) * yolo_grid_width)
                centerY = (0.5 * (box.y1 + box.y2)) / (float(self.height) * yolo_grid_height)
                w = np.log((box.x2 - box.x1) / float(max_anchor.x2))
                h = np.log((box.y2 - box.y1) / float(max_anchor.y2))
                yolobox = [centerX, centerY, w, h]
                anno_idx = list(self.labels).index(box.label)
                gridX = int(np.floor(centerX))
                gridY = int(np.floor(centerY))

                yolo_out[img_count, gridY, gridX, max_index%3] = 0
                yolo_out[img_count, gridY, gridX, max_index%3, 0:4] = yolobox
                yolo_out[img_count, gridY, gridX, max_index%3, 4] = 1.
                yolo_out[img_count, gridY, gridX, max_index%3, 5+anno_idx] = 1

                true_box = [centerX, centerY, box.x2 - box.x1, box.y2 - box.y1]
                groundtruths[img_count, 0, 0, 0, true_box_idx] = true_box
                true_box_idx += 1
                true_box_idx = true_box_idx % self.max_boxes
            input_images[img_count] = img/255
            img_count += 1
        return [input_images, groundtruths, yolo_smallout, yolo_midout, yolo_bigout], [yolo_loss1, yolo_loss2, yolo_loss3]    

    def _get_best_anchor(self, boundbox):
        '''Compare bounding box with all anchors and find best match'''
        max_anchor = None
        max_index = -1
        max_iou = -1
        bb = ia.BoundingBox(x1=0.0, y1=0.0, x2=boundbox.x2, y2=boundbox.y2)
        for idx in range(len(self.anchors)):
            anchor = self.anchors[idx]
            iou = bb.iou(anchor)
            if max_iou < iou:
                max_anchor = anchor
                max_index = idx
                max_iou = iou
        return max_anchor, max_index

#----------------------------------------------
#%%
#----------Callback----------
def create_callbacks():
    return [
        EarlyStopping(
            monitor='loss',
            mode='auto',
            min_delta=0.01,
            patience=7,
            verbose=1),
        ReduceLROnPlateau(
            monitor='loss',
            patience=7,
            verbose=1,
            mode='auto',
            min_delta=0.01),
    ]
#----------------------------
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
        # self.anchors=anchors
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
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.threshold), 4)

        batch_seen = tf.assign_add(tf.Variable(0.), 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, 1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale
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
    if skip:
        return add([skip_connection, x])
    else:
        return x

def YoloV3(numcls,anchors, max_grid, batch_size, threshold, max_boxes):
    img = Input(shape=(None,None,3))
    true_boxes = Input(shape=(1,1,1,max_boxes, 4))
    true_box_1 = Input(shape=(None, None, len(anchors)//6, 5+numcls))
    true_box_2 = Input(shape=(None, None, len(anchors)//6, 5+numcls))
    true_box_3 = Input(shape=(None, None, len(anchors)//6, 5+numcls))
    x = createYoloLyr(img, [
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
            {'filters': 512, 'kernel_size': 3, 'strides': 1, 'bnorm': True, 'leakyRelu': True}])
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
        {'filters': (3*(5 + numcls)), 'kernel_size': 1, 'strides': 1, 'bnorm': False, 'leakyRelu': False}], skip=False)
    loss_small = YoloLossLayer(anchors=anchors[12:],
                                max_grid=[1*num for num in max_grid],
                                batch_size=batch_size,
                                threshold=threshold)([img, smallPred, true_box_1, true_boxes])
    x = createYoloLyr(x, [{'filters': 256, 'kernel_size': 1, 'strides': 1, 'bnorm': False, 'leakyRelu': False}], skip=False)
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
    loss_mid = YoloLossLayer(anchors=anchors[6:12],
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
    loss_big = YoloLossLayer(anchors=anchors[:6],
                            max_grid=[4*num for num in max_grid],
                            batch_size=batch_size,
                            threshold=threshold)([img, bigPred, true_box_3, true_boxes])
    trainModel = Model([img, true_boxes, true_box_1, true_box_2, true_box_3], [loss_small, loss_mid, loss_big])
    inferModel = Model([img, smallPred, midPred, bigPred])
    return [trainModel, inferModel]
#---------------------------------

