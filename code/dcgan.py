import sys
import os
import timeit
import logging

import numpy as np
import theano
import theano.tensor as T

import lasagne

from models import build_discriminator, build_generator

from common import load_data_dcga, shared_dataset

def main(n_epochs=200, learning_rate=0.0009, batch_size=64, num_train=None, dumpIntraining=True, ae_weight=1):

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    filename=os.path.join(os.path.split(__file__)[0],'dump_dcgan.log'))
    # create logger with 'model'
    logger = logging.getLogger('model')
    logger.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # add the handlers to the logger   
    logger.addHandler(ch)


    # Load the dataset
    print("Loading data...")
    datasets = load_data_dcga(num_train)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size

    x_gpu_set, y_gpu_set = shared_dataset(shapeX=(5 * batch_size, 3, 64, 64), shapeY=(5 * batch_size, 3, 64, 64))


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
    y = T.tensor4('y', dtype=theano.config.floatX)  # the labels are presented as rasterized images as well

    # Create neural network model
    print("Building model and compiling functions...")
    generator , ae = build_generator(x)
    discriminator = build_discriminator(y)

    # Create expression for passing real data through the discriminator
    real_out = lasagne.layers.get_output(discriminator)
    # Create expression for passing fake data through the discriminator
    fake_out = lasagne.layers.get_output(discriminator, lasagne.layers.get_output(generator))
    
    # Create loss expressions
    g_cost_d = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()

    discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, 1)
            + lasagne.objectives.binary_crossentropy(fake_out, 0)).mean()

    auto_encoder_loss = lasagne.objectives.squared_error(T.flatten(lasagne.layers.get_output(generator), 2) , T.flatten(y, 2)).mean()

    generator_loss = g_cost_d + auto_encoder_loss / ae_weight
    
    # Create update expressions for training
    generator_params = lasagne.layers.get_all_params(generator, trainable=True)

    discriminator_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    updates = lasagne.updates.adam(
            generator_loss, generator_params, learning_rate=learning_rate)
    updates.update(lasagne.updates.adam(
            discriminator_loss, discriminator_params, learning_rate=learning_rate))

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([index],
                               [(real_out > .5).mean(),
                                (fake_out < .5).mean()],
                                updates=updates,
                                givens={
                                    x: x_gpu_set[index * batch_size: (index + 1) * batch_size],
                                    y: y_gpu_set[index * batch_size: (index + 1) * batch_size]
                                })

    val_fn = theano.function([index],
                               [(real_out > .5).mean(),
                                (fake_out < .5).mean()],
                                givens={
                                    x: x_gpu_set[index * batch_size: (index + 1) * batch_size],
                                    y: y_gpu_set[index * batch_size: (index + 1) * batch_size]
                                })

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
                    np.savez('dcgan_gen.npz', *lasagne.layers.get_all_param_values(generator))
                    np.savez('dcgan_disc.npz', *lasagne.layers.get_all_param_values(discriminator))
                    
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

if __name__ == '__main__':
    num_train = None
    if len(sys.argv) == 2:
        num_train = int(sys.argv[1])
    main(dumpIntraining=True, num_train=num_train)