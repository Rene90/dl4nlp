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
from blocks.bricks import Linear, Softmax, Tanh
from blocks.bricks.lookup import LookupTable

from dataset import createDataset
train_data,vocab_size = createDataset()

def initLayers(layers):
    for l in layers: l.initialize()

HIDDEN_DIM = 100
VOCAB_DIM = vocab_size

# input
x = tensor.ivector('inchar')
y = tensor.imatrix('outchar')

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
    activation = Tanh(),
    weights_init = initialization.IsotropicGaussian(0.01)
)
#
S = Linear(
    input_dim = HIDDEN_DIM,
    output_dim = VOCAB_DIM,
    weights_init = initialization.IsotropicGaussian(0.01),
    biases_init = initialization.Constant(0)
)

A = Softmax()

initLayers([W,H,S])
y_hat = A.apply(S.apply(H.apply(W.apply(x))))
cost = softmax.categorical_cross_entropy(y, y_hat).mean()

main_loop = MainLoop(
    data_stream = DataStream(
        train_data
    ),
    algorithm   = GradientDescent(
        cost=cost,
        parameters=cg.parameters,
        step_rule=Scale(learning_rate=0.1)),
        extensions = [
            DataStreamMonitoring(variables=[cost]),
            FinishAfter(after_n_epochs=10),
            Printing(),
            #TrainingDataMonitoring([cost,], after_batch=True),
        ]
)
main_loop.run()
#f = theano.function([x], out, updates=...)
