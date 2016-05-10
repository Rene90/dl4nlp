#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import numpy
import numpy as np
import theano
from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Linear, Tanh, Rectifier, NDimensionalSoftmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.cost import CategoricalCrossEntropy

from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT

from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint

from dataset import Corpus, createDataset
corpus = Corpus(open("corpus.txt").read())
train_data,vocab_size = createDataset(
    corpus=corpus,
    sequence_length=750,
    repeat=20
)

def initLayers(layers):
    for l in layers: l.initialize()

SAVE_PATH = "./model_checkpoints/savepoint.pkl"
HIDDEN_DIM = 200
VOCAB_DIM = vocab_size
#print "Vocabulary size of", vocab_size

# input
x = tensor.imatrix('inchar')
y = tensor.imatrix('outchar')

#
W = LookupTable(
    name = "W1",
    dim = HIDDEN_DIM,
    length = VOCAB_DIM,
    weights_init = initialization.IsotropicGaussian(0.01),
    biases_init = initialization.Constant(0)
)
# recurrent history weight
H = SimpleRecurrent(
    name = "H",
    dim = HIDDEN_DIM,
    activation = Rectifier(),
    weights_init = initialization.IsotropicGaussian(0.01)
)
#
S = Linear(
    name = "W2",
    input_dim = HIDDEN_DIM,
    output_dim = VOCAB_DIM,
    weights_init = initialization.IsotropicGaussian(0.01),
    biases_init = initialization.Constant(0)
)

A = NDimensionalSoftmax(
    name = "softmax"
)

initLayers([W,H,S])
activations = W.apply(x)
hiddens = H.apply(activations)
activations2 = S.apply(hiddens)
y_hat = A.apply(activations2, extra_ndim=1)
cost = A.categorical_cross_entropy(y, activations2, extra_ndim=1).mean()

cg = ComputationGraph(cost)
#print VariableFilter(roles=[WEIGHT])(cg.variables)
W1,H,W2 = VariableFilter(roles=[WEIGHT])(cg.variables)


if __name__ == "__main__":

    main_loop = MainLoop(
        data_stream = DataStream(
        train_data,
        iteration_scheme = SequentialScheme(
            train_data.num_examples,
            batch_size = 50
        )
    ),
    algorithm = GradientDescent(
        cost  = cost,
        parameters = cg.parameters,
        step_rule = Adam()
    ),
    extensions = [
        #DataStreamMonitoring(variables=[cost]),
        FinishAfter(),
        Printing(),
        ProgressBar(),
        TrainingDataMonitoring([cost,], after_batch=True),
        Checkpoint(SAVE_PATH, every_n_epochs=1),
    ],
    model = Model(y_hat)
)
    main_loop.run()
if _reload_:
    # test
    from blocks.extensions.saveload import load
    main_loop = load(SAVE_PATH)
    main_loop.run()
