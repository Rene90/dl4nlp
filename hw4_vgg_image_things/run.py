#!/usr/bin/python
#
# author: rknaebel
#
# date:
# description:
#
import theano
from theano import tensor
import h5py

from stober.imagenet import ImagenetModel
MODEL_PATH = "./data/imagenet-vgg-verydeep-16.mat"
DATA_PATH = "/projects/korpora/mscoco/coco.hdf5"

mscoco_data = h5py.File(DATA_PATH)

image_model = ImagenetModel(MODEL_PATH)

x = tensor.tensor4("input", dtype="float32")
#y = tensor.imatrix("output")

tmp = x
for layer in image_model.layers[:-1]:
    tmp = layer.apply(tmp)

y_hat = image_model.layers[-1].apply(tmp)
#cost  = image_model.layer[-1].

predict = theano.function([x], y_hat, allow_input_downcast=True)

res = predict(mscoco_data)
