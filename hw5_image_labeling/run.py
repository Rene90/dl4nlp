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
import theano
from theano import tensor
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
from blocks.bricks import Tanh

from blocks.main_loop import MainLoop
from blocks.extensions.saveload import Checkpoint #, load
from blocks.graph import ComputationGraph
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.extensions import FinishAfter, Printing, ProgressBar
#from blocks.extensions.saveload import load
from blocks.serialization import load
from blocks.monitoring import aggregation # ???

from blocks.select import Selector
########################################################################
from stober.imagenet import ImagenetModel
from MySimpleRecurrent import MySimpleRecurrent

from rnn_argparse import getArguments
args = getArguments()

########################################################################
# LOAD DATA
########################################################################
MSCOCO = h5py.File("/projects/korpora/mscoco/coco.hdf5", "r")
LABELS = json.load(open("/projects/korpora/mscoco/coco/cocotalk.json", "r"))

########################################################################
# DATASET
########################################################################
from fuel.datasets.base import Dataset

class ImageLabelingDataset(Dataset):
    def __init__(self, **kwargs):
        self.provides_sources = ('image', 'label')
         # for technical reasons
        self.axis_labels = None
        self.images = MSCOCO["image"]
        self.labels = MSCOCO["sequence"]
        self.vocab_size = max(map(int, LABELS["ix_to_word"].keys()))
        super(ImageLabelingDataset, self).__init__(**kwargs)
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_input_dim(self):
        return self.images.shape[1]
    
    def get_data(self, state=None, request=None):
        return self.images[request], self.labels[request]

########################################################################
# AUXILIARIES
########################################################################
def idx2label(txt):
    " ".join(LABELS["ix_to_word"].get(str(w), "") for w in txt)

########################################################################
# NN DEFs
########################################################################

def createConvNetFn():
    """
    Returns the prediction function of the conv net which can be used
    separately to transform an image into either class probs or one level
    earlier returning the last hidden activation (might have more value)
    """
    IMG_MODEL = ImagenetModel("/projects/korpora/mscoco/coco/imagenet-vgg-verydeep-16.mat")
    x = tensor.tensor4("input", dtype="float32")
    y_hat = x
    for layer in IMG_MODEL.layers[:-1]:
        y_hat = layer.apply(y_hat)
    predict = theano.function([x], y_hat, allow_input_downcast=True)
    return predict

def applyConvNet(x):
    """
    Load the convnet given the different layers defined in ImageModel
    Apply the network to the given input and return the so applied net
    """
    IMG_MODEL = ImagenetModel("/projects/korpora/mscoco/coco/imagenet-vgg-verydeep-16.mat")
    tmp = x
    for layer in IMG_MODEL.layers[:-1]:
        tmp = layer.apply(tmp)
    y_hat = IMG_MODEL.layers[-1].apply(tmp)
    return y_hat

def getRnnGenerator(vocab_size,hidden_dim,input_dim=512):
    """
    "Apply" the RNN to the input x
    For initializing the network, the vocab size needs to be known
    Default of the hidden layer is set tot 512 like Karpathy
    """
    generator = SequenceGenerator(
        Readout(readout_dim = vocab_size,
                source_names = ["states"], # transition.apply.states ???
                emitter = SoftmaxEmitter(name="emitter"),
                feedback_brick = LookupFeedback(
                    vocab_size,
                    input_dim,
                    name = 'feedback'
                ),
                name = "readout"
        ),
        MySimpleRecurrent(
            name = "transition",
            activation = Tanh(),
            dim = hidden_dim
        ),
        weights_init = IsotropicGaussian(0.01),
        biases_init  = Constant(0),
        name = "generator"
    )
    generator.push_initialization_config()
    generator.transition.weights_init = IsotropicGaussian(0.01)
    generator.initialize()
    
    return generator

########################################################################
# ???
########################################################################


########################################################################
# Main
########################################################################
if __name__ == "__main__":
    dataset = ImageLabelingDataset()
    x = tensor.matrix("image")
    y = tensor.lmatrix("label")
    vocab_size = dataset.get_vocab_size()
    #x = tensor.matrix("input", dtype="float32")
    #image_act_fn = createConvNetFn()
    #y_hat = applyRNN(x,vocab_size,image_act_fn)
    hidden_dim = dataset.get_input_dim()
    rnn = getRnnGenerator(vocab_size,hidden_dim)
    cost = rnn.cost(y, context=x)
    cost.name = "sequence_cost"
    print "initialized..."
    algorithm = GradientDescent(
        cost = cost,
        parameters = list(Selector(rnn).get_parameters().values()),
        step_rule = Adam(),
        # because we want use all the stuff in the training data
        on_unused_sources = 'ignore'
    )
    main_loop = MainLoop(
        algorithm=algorithm,
        data_stream = DataStream(
            dataset,
            iteration_scheme = SequentialScheme(
                dataset.num_examples,
                batch_size = 20
            )
        ),
        model=Model(cost),
        extensions=[
            FinishAfter(),
            TrainingDataMonitoring([cost], prefix="this_step",
                                           after_batch=True),
            TrainingDataMonitoring([cost], prefix="average",
                                           every_n_batches=100),
            Checkpoint(args.model, every_n_epochs=5),
            Printing(every_n_batches=100)])
    print "start training"
    main_loop.run()
