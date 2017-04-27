import os
import numpy as np
import theano
import theano.tensor as T
from common import load_data_predict
from models import build_convAutoencoder as cnn

import PIL.Image as Image
import lasagne

def predict(model="cnn"):

    valid_set_x, valid_set_y = load_data_predict(70)

    # start-snippet-1
    x = T.tensor4('x', dtype=theano.config.floatX)   # the data is presented as rasterized images
    y = T.tensor4('y', dtype=theano.config.floatX)  # the labels are presented as rasterized images as well

    if model == 'mlp':
        pass
    elif model == 'cnn':
        network = cnn(x)

    with np.load(os.path.join(os.path.split(__file__)[0], "model.npz")) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    prediction = lasagne.layers.get_output(network)

    forward = theano.function(
        [x],
        prediction
    )

    p = forward(valid_set_x)

    count = p.shape[0]

    center = (int(np.floor(valid_set_y.shape[2] / 2.)), int(np.floor(valid_set_y.shape[3] / 2.)))

    for i in range(count):
        generated = np.transpose(p[i].astype(np.uint8), (1, 2, 0))
        org_img = np.transpose(valid_set_y[i].astype(np.uint8), (1, 2, 0))
        tobe_img = np.transpose(valid_set_x[i].astype(np.uint8), (1, 2, 0))
        tobe_img[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = generated
        img_tobe = Image.fromarray(tobe_img)
        img_org = Image.fromarray(org_img)
        img_tobe.show()
        img_org.show()
        img_tobe.save(os.path.join(os.path.split(__file__)[0], "..", "data", "prediction", i + "-tobe" + ".jpg"))
        img_org.save(os.path.join(os.path.split(__file__)[0], "..", "data", "prediction", i + "-org" + ".jpg"))
        
if __name__ == '__main__':
    predict()