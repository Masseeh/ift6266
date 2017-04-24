import os, sys
import glob
import six.moves.cPickle as pkl
import numpy as np
import PIL.Image as Image

import theano


def load_data(test_num = np.inf):

    train_data_path = os.path.join(os.path.split(__file__)[0], "..", "data", "inpainting", "train2014")
    val_data_path = os.path.join(os.path.split(__file__)[0], "..", "data", "inpainting", "val2014")
    
    #Training set

    print(train_data_path + "/*.jpg")
    train_imgs = glob.glob(train_data_path + "/*.jpg")

    train_set_x = []
    train_set_y = []

    for i, img_path in enumerate(train_imgs):
        if i > test_num:
            break
        img = Image.open(img_path)
        img_array = np.array(img)

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
            train_set_x.append(input)      
            train_set_y.append(img_array)
              


    #Validation set

    print(val_data_path + "/*.jpg")
    
    val_imgs = glob.glob(val_data_path + "/*.jpg")   
    val_set_x = []
    val_set_y = []

    for i, img_path in enumerate(val_imgs):
        if i > test_num:
            break
        img = Image.open(img_path)
        img_array = np.array(img)

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
            val_set_x.append(input)
            val_set_y.append(img_array)     
            


        # Image.fromarray(img_array).show()
        # Image.fromarray(input).show()
        # Image.fromarray(target).show()
        # print(i, caption_dict[cap_id])

    def shared_dataset(data_x , data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_y

    valid_shared_set_x, valid_shared_set_y = shared_dataset(val_set_x, val_set_y)
    train_shared_set_x, train_shared_set_y = shared_dataset(train_set_x, train_set_y)

    rval = [(train_shared_set_x, train_shared_set_y), (valid_shared_set_x, valid_shared_set_y)]

    return rval

if __name__ == '__main__':
    load_data(3)
