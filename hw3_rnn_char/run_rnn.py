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
from blocks.bricks import Identity
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Linear
from blocks.bricks.lookup import LookupTable



# input
x = tensor.tensor3('inchar')
y = tensor.tensor('outchar')

# recurrent history weight
H = SimpleRecurrent(
    dim=3,
    activation=Tanh(),
    weights_init=initialization.IsotropicGaussian(0.01)
)
#
W = LookupTable(
    dim = 3,
    length=VOCAB_DIM
    weights_init = initialization.IsotropicGaussian(0.01),
    biases_init = initialization.Constant(0)
)

S = Linear(
    input_dim = HIDDEN_DIM,
    output_dim = VOCAB_DIM
)

H.initialize()
W.initialize()

out = H.apply(W.apply(x))

cost = softmax.categorical_cross_entropy(y, y_hat, extra_ndim=1).mean()

f = theano.function([x], out, updates=...)
