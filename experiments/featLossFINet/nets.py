"""
net works
"""
import os
from os import path
import argparse
import random
import csv
from tqdm import tqdm
import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import h5py
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import (reporter, training)
from chainer.training import extensions
from chainer.datasets import (TupleDataset, TransformDataset)
from chainer.links.model.vision import resnet
from chainercv import transforms

class VGG16(chainer.Chain):
    '''
    特徴抽出としてのVGG16
    '''
    def __init__(self, pooling=F.max_pooling_2d):
        super(VGG16, self).__init__()
        self.pooling = pooling

        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)

    def forward(self, x):
        # 1 Layer
        h  = F.relu(self.conv1_1(x))
        h1 = F.relu(self.conv1_2(h))
        # 2 Layer
        h  = self.pooling(h1, ksize=2)
        h  = F.relu(self.conv2_1(h))
        h2 = F.relu(self.conv2_2(h))
        # 3 Layer
        h = self.pooling(h2, ksize=2)
        h  = F.relu(self.conv3_1(h))
        h  = F.relu(self.conv3_2(h))
        h3 = F.relu(self.conv3_3(h))
        # 4 Layer
        h = self.pooling(h3, ksize=2)
        h  = F.relu(self.conv4_1(h))
        h  = F.relu(self.conv4_2(h))
        h4 = F.relu(self.conv4_3(h))
        # 5 Layer
        h = self.pooling(h4, ksize=2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h5 = F.relu(self.conv5_3(h))
        return h1, h2, h3, h4, h5

class Unet(chainer.Chain):
    def __init__(self, ch=8):
        init_w = chainer.initializers.HeNormal()
        super(Unet, self).__init__()

        with self.init_scope():
            # encoder
            # 64
            self.enc_conv0 = L.Convolution2D(None, ch * 1, 3, 1, 1, initialW=init_w)
            self.enc_conv1 = L.Convolution2D(ch * 1, ch * 1, 3, 1, 1, initialW=init_w)
            # 32
            self.enc_conv2 = L.Convolution2D(ch * 1, ch * 2, 3, 1, 1, initialW=init_w)
            self.enc_conv3 = L.Convolution2D(ch * 2, ch * 2, 3, 1, 1, initialW=init_w)
            # 16
            self.enc_conv4 = L.Convolution2D(ch * 2, ch * 4, 3, 1, 1, initialW=init_w)
            self.enc_conv5 = L.Convolution2D(ch * 4, ch * 4, 3, 1, 1, initialW=init_w)
            # 8
            self.enc_conv6 = L.Convolution2D(ch * 4, ch * 8, 3, 1, 1, initialW=init_w)

            #decoder
            # 8-> 16
            self.dec_conv0 = L.Convolution2D(ch * 8, ch * 8, 3, 1, 1, initialW=init_w)
            self.dec_deconv1 = L.Convolution2D(ch * 8, ch * 4, 3, 1, 1, initialW=init_w)
            # 16 -> 32
            self.dec_conv2 = L.Convolution2D(ch * 8, ch * 4, 3, 1, 1, initialW=init_w)
            self.dec_conv3 = L.Convolution2D(ch * 4, ch * 4, 3, 1, 1, initialW=init_w)
            self.dec_deconv4 = L.Convolution2D(ch * 4, ch * 2, 3, 1, 1, initialW=init_w)
            # 32 -> 64
            self.dec_conv5 = L.Convolution2D(ch * 4, ch * 2, 3, 1, 1, initialW=init_w)
            self.dec_conv6 = L.Convolution2D(ch * 2, ch * 2, 3, 1, 1, initialW=init_w)
            self.dec_deconv7 = L.Convolution2D(ch * 2, ch * 1, 3, 1, 1, initialW=init_w)

            self.conv8 = L.Convolution2D(ch * 2, ch, 3, 1, 1, initialW=init_w)
            self.conv9 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=init_w)
            self.conv10 = L.Convolution2D(ch, 3, 1, 0, 0, initialW=init_w)

    def forward(self, x):
        # encode
        batch, f, c, w, h = x.shape
        h = F.reshape(x, (batch, f * c, w, h))
        # 64
        h = F.relu(self.enc_conv0(h))
        h0 = F.relu(self.enc_conv1(h))
        # 32
        h = F.max_pooling_2d(h0, 2)
        h = F.relu(self.enc_conv2(h))
        h1 = F.relu(self.enc_conv3(h))
        # 16
        h = F.max_pooling_2d(h1, 2)
        h = F.relu(self.enc_conv4(h))
        h2 = F.relu(self.enc_conv5(h))
        # 8
        h = F.max_pooling_2d(h2, 2)
        h = F.relu(self.enc_conv6(h))

        # decode
        # 8 -> 16
        h = F.relu(self.dec_conv0(h))
        h = F.concat((F.relu(self.dec_deconv1(h)), h2), axis=1)
        # 16 -> 32
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        h = F.concat((F.relu(self.dec_deconv4(h)), h1), axis=1)
        # 32 -> 64
        h = F.relu(self.dec_conv5(h))
        h = F.relu(self.dec_conv6(h))
        h = F.concat((F.relu(self.dec_deconv7(h)), h0), axis=1)

        # gen image
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = self.conv10(h)
        return h

    def get_loss_func(self, C=1.0):
        def lf(x, t):
            y = self.forward(x)
            mse_loss = F.mean_squared_error(y, t)
