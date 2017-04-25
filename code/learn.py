import sys
import os
import timeit

import numpy as np
import theano
import theano.tensor as T
from common import load_data
from models import build_convAutoencoder as cnn

import lasagne

def main(model='cnn',learning_rate=0.1, n_epochs=200, batch_size=500, dumpIntraining=False):

    # Load the dataset
    print("Loading data...")

    datasets = load_data(10000)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.tensor4('x')   # the data is presented as rasterized images
    y = T.tensor4('y')  # the labels are presented as rasterized images as well

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        pass
    elif model == 'cnn':
        network = cnn(x)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (L2 error):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, y)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

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
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(
        [index],
        loss,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
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
        train_losses = 0
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_fn(minibatch_index)

            train_losses += cost_ij

            if (iter + 1) % validation_frequency == 0:

                this_train_loss = train_losses/n_train_batches
                # compute zero-one loss on validation set
                validation_losses = [val_fn(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, training error %f %%, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_train_loss * 100.,
                       this_validation_loss * 100.))
                
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

                    # test it on the test set
                    # test_losses = [
                    #     test_model(i)
                    #     for i in range(n_test_batches)
                    # ]
                    # test_score = numpy.mean(test_losses)
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #        'best model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, ' %
          (best_validation_loss * 100., best_iter + 1))
    print(('The code for file ' +
           os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    # Dump the network weights to a file:
    np.savez(os.path.join(os.path.split(__file__)[0], 'model.npz'), *lasagne.layers.get_all_param_values(network))
    
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    
if __name__ == '__main__':
    main()