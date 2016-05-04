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

from dataset import createDataset
train_data,vocab_size = createDataset()

def initLayers(layers):
    for l in layers: l.initialize()

HIDDEN_DIM = 100
VOCAB_DIM = vocab_size

# input
x = tensor.imatrix('inchar')
y = tensor.tensor3('outchar')

#
W = LookupTable(
    dim = HIDDEN_DIM,
    length = VOCAB_DIM,
    weights_init = initialization.IsotropicGaussian(0.01),
    biases_init = initialization.Constant(0)
)
# recurrent history weight
H = SimpleRecurrent(
    dim = HIDDEN_DIM,
    activation = Rectifier(),
    weights_init = initialization.IsotropicGaussian(0.01)
)
#
S = Linear(
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
y_hat = A.apply(activations2)
cost = CategoricalCrossEntropy(y, y_hat, extra_ndim=1).mean()

"""
main_loop = MainLoop(
    data_stream = DataStream(
        train_data
    ),
    algorithm = GradientDescent(
        cost  = cost,
        parameters = cg.parameters,
        step_rule = Scale(learning_rate=0.1)),
        extensions = [
            DataStreamMonitoring(variables=[cost]),
            FinishAfter(after_n_epochs=10),
            Printing(),
            #TrainingDataMonitoring([cost,], after_batch=True),
        ]
)
main_loop.run()
"""
#f = theano.function([x], out, updates=...)
