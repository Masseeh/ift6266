from __future__ import print_function
import sys
import os
import timeit
import numpy as np
import theano
import theano.tensor as T
from common import load_data, shared_dataset
from models import build_convAutoencoder as cnn
import logging

import lasagne

def main(model='cnn',learning_rate=0.0009, n_epochs=200, batch_size=64, dumpIntraining=False, num_train=None):

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filename=os.path.join(os.path.split(__file__)[0],'dump.log'))
    # create logger with 'model'
    logger = logging.getLogger('model')
    logger.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # add the handlers to the logger   
    logger.addHandler(ch)

    # Load the dataset
    # print("Loading data...")
    logger.info("Loading data...")

    datasets = load_data(num_train)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size

    x_gpu_set, y_gpu_set = shared_dataset(shapeX=(5 * batch_size, 3, 64, 64), shapeY=(5 * batch_size, 3, 32, 32))

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
    y = T.tensor4('y', dtype=theano.config.floatX)  # the labels are presented as rasterized images as well

    # Create neural network model (depending on first command line parameter)
    logger.info("Building model and compiling functions...")
    if model == 'mlp':
        pass
    elif model == 'cnn':
        network = cnn(x)
    else:
        logger.info("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (L2 error):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, y)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Adam.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=learning_rate)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function(
        [index],
        loss,
        updates=updates,
        givens={
            x: x_gpu_set[index * batch_size: (index + 1) * batch_size],
            y: y_gpu_set[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(
        [index],
        loss,
        givens={
            x: x_gpu_set[index * batch_size: (index + 1) * batch_size],
            y: y_gpu_set[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    logger.info('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_losses = []
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if minibatch_index % 5 == 0:
                logger.info("epoch %d , load training batch %d into gpu" ,epoch ,minibatch_index)
                x_gpu_set.set_value(train_set_x[minibatch_index * batch_size: (minibatch_index + 5) * batch_size])
                y_gpu_set.set_value(train_set_y[minibatch_index * batch_size: (minibatch_index + 5) * batch_size])

            if iter % 100 == 0:
                logger.info('training iter = %d', iter)

            cost_ij = train_fn(minibatch_index%5)

            train_losses.append(cost_ij)

            if (iter + 1) % validation_frequency == 0:

                this_train_loss = np.mean(train_losses)
                train_losses = []
                # compute loss on validation set
                validation_losses = []

                for val_idx in range(n_valid_batches):

                    if val_idx % 5 == 0:
                        logger.info("epoch %d, load validation batch %d into gpu" ,epoch ,val_idx)
                        x_gpu_set.set_value(valid_set_x[val_idx * batch_size: (val_idx + 5) * batch_size])
                        y_gpu_set.set_value(valid_set_y[val_idx * batch_size: (val_idx + 5) * batch_size])
                    
                    val_cost_ij = val_fn(val_idx%5)

                    validation_losses.append(val_cost_ij)

                this_validation_loss = np.mean(validation_losses)
                logger.info('epoch %i, minibatch %i/%i, training error %f, validation error %f' %
                    (epoch, minibatch_index + 1, n_train_batches, this_train_loss,
                    this_validation_loss))
                
                if dumpIntraining:
                    np.savez(os.path.join(os.path.split(__file__)[0], 'model.npz'), *lasagne.layers.get_all_param_values(network))
                    
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    logger.info('Optimization complete.')
    logger.info('Best validation score of %f %% obtained at iteration %i, ' %
          (best_validation_loss * 100., best_iter + 1))
    logger.info('The code for file ' +
           os.path.split(__file__)[1] + ' ran for %.2fm' ,((end_time - start_time) / 60.))

    # Dump the network weights to a file:
    np.savez(os.path.join(os.path.split(__file__)[0], 'model.npz'), *lasagne.layers.get_all_param_values(network))
    
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    
if __name__ == '__main__':
    num_train = None
    if len(sys.argv) == 2:
        num_train = int(sys.argv[1])
    main(dumpIntraining=True, num_train=num_train)
