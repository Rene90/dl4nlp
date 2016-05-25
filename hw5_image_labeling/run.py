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
from MySimpleRecurrent import MySimpleRecurrent

MSCOCO = h5py.File("/projects/korpora/mscoco/coco/cocotalk.h5", "r")
LABELS = json.load(open("/projects/korpora/mscoco/coco/cocotalk.json", "r"))
IMG_MODEL = ImagenetModel("/projects/korpora/mscoco/coco/imagenet-vgg-verydeep-16.mat")


# Procedures
def idx2label(txt):
    " ".join(LABELS["ix_to_word"].get(str(w), "<unk>") for w in txt)

def applyConvNet(x):
    tmp = x
    for layer in IMG_MODEL.layers[:-1]:
        tmp = layer.apply(tmp)
    y_hat = IMG_MODEL.layers[-1].apply(tmp)
    return y_hat

def applyRNN(x):
    # Build the bricks and initialize them
    transition = GatedRecurrent(name="transition", dim=dim,
                                activation=Tanh())
    generator = SequenceGenerator(
        Readout(readout_dim = vocab_size,
                source_names = ["states"], # transition.apply.states ???
                emitter = SoftmaxEmitter(name = "emitter"),
                feedback_brick = LookupFeedback(
                    vocab_size,
                    feedback_dim,
                    name = 'feedback'
                ),
                name = "readout"),
        transition,
        weights_init = IsotropicGaussian(0.01),
        biases_init  = Constant(0),
        name = "generator"
    )
    generator.push_initialization_config()
    transition.weights_init = Orthogonal()
    generator.initialize()

    # Build the cost computation graph.
    x = tensor.lmatrix('inchar')

    cost = generator.cost(outputs=x)
    cost.name = "sequence_cost"


# Main
if __name__ == "__name__":
    x = tensor.tensor4("input", dtype="float32")
    
