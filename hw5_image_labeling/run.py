#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
########################################################################
# START IMPORT
########################################################################
import h5py
import json
########################################################################
from blocks.initialization import (
    IsotropicGaussian, 
    Constant, 
    Orthogonal #?
)
from blocks.algorithms import (
    GradientDescent, 
    Adam
)
from blocks.bricks.sequence_generators import (
    SequenceGenerator, 
    Readout, 
    SoftmaxEmitter, 
    LookupFeedback
)
from blocks.select import Selector
########################################################################
from stober.imagenet import ImagenetModel
from MySimpleRecurrent import MySimpleRecurrent

try:
    from britta import loadDataset
except ImportError:
    print "Where is the dataset? :("

########################################################################
# LOAD DATA
########################################################################
MSCOCO = h5py.File("/projects/korpora/mscoco/coco/cocotalk.h5", "r")
LABELS = json.load(open("/projects/korpora/mscoco/coco/cocotalk.json", "r"))
IMG_MODEL = ImagenetModel("/projects/korpora/mscoco/coco/imagenet-vgg-verydeep-16.mat")

########################################################################
# AUXILIARIES
########################################################################
def idx2label(txt):
    " ".join(LABELS["ix_to_word"].get(str(w), "<unk>") for w in txt)

########################################################################
# NN DEFs
########################################################################

def applyConvNet(x):
    """
    Load the convnet given the different layers defined in ImageModel
    Apply the network to the given input and return the so applied net
    """
    tmp = x
    for layer in IMG_MODEL.layers[:-1]:
        tmp = layer.apply(tmp)
    y_hat = IMG_MODEL.layers[-1].apply(tmp)
    return y_hat

def applyRNN(x,vocab_size,input_dim=512):
    """
    "Apply" the RNN to the input x
    For initializing the network, the vocab size needs to be known
    Default of the hidden layer is set tot 512 like Karpathy
    """
    transition = MySimpleRecurrent(name="transition", dim=input_dim)
    generator = SequenceGenerator(
        Readout(readout_dim = vocab_size,
                source_names = ["states"], # transition.apply.states ???
                emitter = SoftmaxEmitter(name = "emitter"),
                feedback_brick = LookupFeedback(
                    vocab_size,
                    input_dim,
                    name = 'feedback'
                ),
                name = "readout"),
        transition,
        weights_init = IsotropicGaussian(0.01),
        biases_init  = Constant(0),
        name = "generator"
    )
    generator.push_initialization_config()
    transition.weights_init = IsotropicGaussian(0.01)
    generator.initialize()
    
    return generator

########################################################################
# ???
########################################################################


########################################################################
# Main
########################################################################
if __name__ == "__name__":
    dataset = None
    
    
    x = tensor.tensor4("input", dtype="float32")
    image_activation = applyConvNet(x)
    y_hat = applyRNN(x,
    
