from __future__ import print_function
import os, sys
import glob
import six.moves.cPickle as pkl
import numpy as np
import PIL.Image as Image

import theano
import logging
import lasagne

class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)

def shared_dataset(shapeX, shapeY, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(np.zeros(shapeX, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.zeros(shapeY, dtype=theano.config.floatX), borrow=borrow)

    return shared_x, shared_y

def load_data(test_num, weight=1, offset=0):

    # create logger with 'model'
    logger = logging.getLogger('model')
    
    train_data_path = os.path.join(os.path.split(__file__)[0], "..", "data", "inpainting", "train2014")
    val_data_path = os.path.join(os.path.split(__file__)[0], "..", "data", "inpainting", "val2014")
    
    #Training set

    logger.info(train_data_path + "/*.jpg")
    train_imgs = glob.glob(train_data_path + "/*.jpg")

    train_set_x = []
    train_set_y = []

    for i, img_path in enumerate(train_imgs):
        if test_num is not None and i > test_num:
            break
        img = Image.open(img_path)
        img_array = np.array(img)

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (2, 0, 1))/weight + offset
            input = np.copy(img_array)
            input[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
            target = img_array[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
            train_set_x.append(input)      
            train_set_y.append(target)
              


    #Validation set

    logger.info(val_data_path + "/*.jpg")
    
    val_imgs = glob.glob(val_data_path + "/*.jpg")   
    val_set_x = []
    val_set_y = []

    for i, img_path in enumerate(val_imgs):
        if test_num is not None and i > test_num:
            break
        img = Image.open(img_path)
        img_array = np.array(img)

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (2, 0, 1))/weight + offset
            input = np.copy(img_array)
            input[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
            target = img_array[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
            val_set_x.append(input)
            val_set_y.append(target)     
            

    val_set_x = np.asarray(val_set_x, dtype=theano.config.floatX)
    val_set_y = np.asarray(val_set_y, dtype=theano.config.floatX)
    train_set_x = np.asarray(train_set_x, dtype=theano.config.floatX)
    train_set_y = np.asarray(train_set_y, dtype=theano.config.floatX)

    rval = [(train_set_x, train_set_y), (val_set_x, val_set_y)]

    return rval

def load_data_predict(test_num, weight=1, offset=0):

    val_data_path = os.path.join(os.path.split(__file__)[0], "..", "data", "inpainting", "val2014")
    
    #Training set
    
    val_imgs = glob.glob(val_data_path + "/*.jpg")   
    val_set_x = []
    val_set_y = []

    for i, img_path in enumerate(val_imgs):
        if test_num is not None and i > test_num:
            break
        img = Image.open(img_path)
        img_array = np.array(img)

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (2, 0, 1))/weight + offset
            input = np.copy(img_array)
            input[:, center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
            target = img_array[:, center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
            val_set_x.append(input)
            val_set_y.append(img_array)     

    val_set_x = np.asarray(val_set_x, dtype=theano.config.floatX)
    val_set_y = np.asarray(val_set_y, dtype=theano.config.floatX)

    rval = (val_set_x, val_set_y)

    return rval

if __name__ == '__main__':
    load_data(3)
