#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import numpy
import theano
from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Linear, Tanh, Rectifier, NDimensionalSoftmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.cost import CategoricalCrossEntropy

from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT

from blocks.extensions import FinishAfter, Printing, ProgressBar

from dataset import createDataset
train_data,vocab_size = createDataset()

def initLayers(layers):
    for l in layers: l.initialize()

HIDDEN_DIM = 100
VOCAB_DIM = vocab_size
print "Vocabulary size of", vocab_size

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

A = NDimensionalSoftmax()

initLayers([W,H,S])
activations = W.apply(x)
hiddens = H.apply(activations)
activations2 = S.apply(hiddens)
y_hat = A.apply(activations2, extra_ndim=1)
cost = A.categorical_cross_entropy(y, activations2, extra_ndim=1).mean()

cg = ComputationGraph(cost)
#print VariableFilter(roles=[WEIGHT])(cg.variables)
W1,H,W2 = VariableFilter(roles=[WEIGHT])(cg.variables)

main_loop = MainLoop(
    data_stream = DataStream(
        train_data,
        iteration_scheme = SequentialScheme(
            train_data.num_examples,
            batch_size = 1
        )
    ),
    algorithm = GradientDescent(
        cost  = cost,
        parameters = cg.parameters,
        step_rule = Scale(learning_rate=0.1)
    ),
    extensions = [
        #DataStreamMonitoring(variables=[cost]),
        FinishAfter(after_n_epochs=1),
        Printing(),
        TrainingDataMonitoring([cost,], after_batch=True),
    ]
)
main_loop.run()

#print W1.get_name()
#print W1.get_value(), H.get_value(), W2.get_value()
#print W1.get_value().shape, H.get_value().shape, W2.get_value().shape

#x_test = tensor.ivector('input')
#y_test = tensor.ivector('output')


#f = theano.function([x], y_hat, updates=...)
