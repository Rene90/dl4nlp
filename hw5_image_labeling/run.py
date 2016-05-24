#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
import h5py
import json

from stober.imagenet import ImagenetModel

MSCOCO = h5py.File("/projects/korpora/mscoco/coco/cocotalk.h5", "r")
LABELS = json.load(open("/projects/korpora/mscoco/coco/cocotalk.json", "r"))
IMG_MODEL = ImagenetModel("data/imagenet-vgg-verydeep-16.mat")


# Procedures
def idx2label(txt):
    " ".join(LABELS["ix_to_word"].get(str(w), "<unk>") for w in txt)

def applyConvNet(x):
    tmp = x
    for layer in IMG_MODEL.layers[:-1]:
        tmp = layer.apply(tmp)
    y_hat = IMG_MODEL.layers[-1].apply(tmp)
    return y_hat

# Main
if __name__ == "__name__":
    x = tensor.tensor4("input", dtype="float32")
    
