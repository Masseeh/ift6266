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
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                                input_var=input_var)

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (32-5+1 , 32-5+1) = (28, 28)
        conv_1 = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())

        # maxpooling reduces this further to (28/2, 28/2) = (14, 14) 
        # 4D output tensor is thus of shape (batch_size, 32, 14, 14)        
        network = lasagne.layers.MaxPool2DLayer(conv_1, pool_size=(2, 2))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (14-5+1, 14-5+1) = (10, 10)
        conv_2 = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify)

        # maxpooling reduces this further to (10/2, 10/2) = (5, 5)
        # 4D output tensor is thus of shape (batch_size, 32, 5, 5)
        network = lasagne.layers.MaxPool2DLayer(conv_2, pool_size=(2, 2))

        # A fully-connected layer of 500 units:
        network = lasagne.layers.DenseLayer(
                network,
                num_units=500,
                nonlinearity=lasagne.nonlinearities.rectify)

        # reshape the layer with 20 kernels
        network = lasagne.layers.ReshapeLayer(network, ([0], 20, 5, 5))

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