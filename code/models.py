from __future__ import print_function
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import batch_norm
from common import Deconv2DLayer

def build_discriminator(input_var=None):

    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

    layer = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    
    # 64x16x16
    layer = batch_norm(lasagne.layers.Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))

    # 64x8x8
    layer = batch_norm(lasagne.layers.Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))

    # 64x4x4
    layer = batch_norm(lasagne.layers.Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))

    # output layer
    layer = lasagne.layers.DenseLayer(layer, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
    print ("Discriminator output:", layer.output_shape)
    return layer


def build_generator(input_var=None):

        # Input layer, as usual:
        ae = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=input_var)

        # 64x32x32
        ae = batch_norm(lasagne.layers.Conv2DLayer(ae, 64, 5, stride=2, pad=2))

        # 64x16x16
        ae = batch_norm(lasagne.layers.Conv2DLayer(ae, 64, 5, stride=2, pad=2))

        # 64x8x8
        ae = batch_norm(lasagne.layers.Conv2DLayer(ae, 64, 5, stride=2, pad=2))

        # 128x4x4
        ae = batch_norm(lasagne.layers.Conv2DLayer(ae, 128, 5, stride=2, pad=2))

        # 64x8x8
        network = batch_norm(Deconv2DLayer(ae, 64, 5, stride=2, pad=2))

        # 64x16x16
        network = batch_norm(Deconv2DLayer(network, 64, 5, stride=2, pad=2))

        # 3x32x32
        network = Deconv2DLayer(network, 3, 5, stride=2, pad=2, nonlinearity=lasagne.nonlinearities.tanh)

        print ("Generator output:", network.output_shape)

        return network

def build_convAutoencoder(input_var=None):

        # As a third model, we'll create a CNN of two convolution + pooling stages
        # and a fully-connected hidden layer in front of the output layer.

        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                                input_var=input_var, name="input")

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (64-5+1 , 64-5+1) = (60, 60)
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5))

        # maxpooling reduces this further to (60/2, 60/2) = (30, 30) 
        # 4D output tensor is thus of shape (batch_size, 32, 30, 30)        
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (30-5+1, 30-5+1) = (26, 26)
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5))

        # maxpooling reduces this further to (26/2, 26/2) = (13, 13)
        # 4D output tensor is thus of shape (batch_size, 32, 13, 13)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Construct the third convolutional pooling layer
        # filtering reduces the image size to (13-4+1, 13-4+1) = (10, 10)
        conv_3 = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(4, 4))

        # maxpooling reduces this further to (10/2, 10/2) = (5, 5)
        # 4D output tensor is thus of shape (batch_size, 32, 5, 5)
        network = lasagne.layers.MaxPool2DLayer(conv_3, pool_size=(2, 2))

        # A fully-connected layer of 800 units:
        network = lasagne.layers.DenseLayer(
                network,
                num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify)

        # reshape the layer with 20 kernels
        network = lasagne.layers.ReshapeLayer(network, ([0], 32, 5, 5))

        #UnPooled layer 32x10x10
        network = lasagne.layers.Upscale2DLayer(network, 2)

        # Construct the first deconvolutional 32x14x14
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5), pad='full')

        #UnPooled layer 32x28x28
        network = lasagne.layers.Upscale2DLayer(network, 2)

        # Construct the first deconvolutional 3x32x32
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=3, filter_size=(5, 5), pad='full')

        return network

if __name__ == '__main__':
#     build_convAutoencoder()
        build_generator()