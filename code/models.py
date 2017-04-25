import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

class Unpool2DLayer(lasagne.layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)



def build_convAutoencoder(input_var=None):

        # As a third model, we'll create a CNN of two convolution + pooling stages
        # and a fully-connected hidden layer in front of the output layer.

        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                                input_var=input_var, name="input")

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (64-5+1 , 64-5+1) = (60, 60)
        conv_1 = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform(), name="conv1")

        # maxpooling reduces this further to (60/2, 60/2) = (30, 30) 
        # 4D output tensor is thus of shape (batch_size, 32, 30, 30)        
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), name="pool1")

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (30-5+1, 30-5+1) = (26, 26)
        conv_2 = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify, name="conv2")

        # maxpooling reduces this further to (26/2, 26/2) = (13, 13)
        # 4D output tensor is thus of shape (batch_size, 32, 13, 13)
        network = lasagne.layers.MaxPool2DLayer(conv_2, pool_size=(2, 2))

        # Construct the third convolutional pooling layer
        # filtering reduces the image size to (13-4+1, 13-4+1) = (10, 10)
        conv_3 = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(4, 4),
                nonlinearity=lasagne.nonlinearities.rectify)

        # maxpooling reduces this further to (10/2, 10/2) = (5, 5)
        # 4D output tensor is thus of shape (batch_size, 32, 5, 5)
        network = lasagne.layers.MaxPool2DLayer(conv_3, pool_size=(2, 2))

        # A fully-connected layer of 500 units:
        network = lasagne.layers.DenseLayer(
                network,
                num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify)

        # reshape the layer with 20 kernels
        network = lasagne.layers.ReshapeLayer(network, ([0], 32, 5, 5))

        #UnPooled layer
        network = Unpool2DLayer(network, ds=(2, 2))

        # Construct the first deconvolutional
        network = lasagne.layers.Deconv2DLayer(conv_3, conv_3.input_shape[1],
                conv_3.filter_size, stride=conv_3.stride, crop=conv_3.pad,
                W=conv_3.W, flip_filters=not conv_3.flip_filters)

        #UnPooled layer
        network = Unpool2DLayer(network, ds=(2, 2))

        # Construct the first deconvolutional
        network = lasagne.layers.Deconv2DLayer(conv_2, conv_2.input_shape[1],
                conv_2.filter_size, stride=conv_2.stride, crop=conv_2.pad,
                W=conv_2.W, flip_filters=not conv_2.flip_filters)

        #UnPooled layer
        network = Unpool2DLayer(network, ds=(2, 2))

        # Construct the second deconvolutional
        network = lasagne.layers.Deconv2DLayer(conv_1, conv_1.input_shape[1],
                conv_1.filter_size, stride=conv_1.stride, crop=conv_1.pad,
                W=conv_1.W, flip_filters=not conv_1.flip_filters)

        return network

if __name__ == '__main__':
    build_convAutoencoder()