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


# input
x = tensor.tensor3('x')

H = SimpleRecurrent(
    dim=3, activation=Identity(), weights_init=initialization.Identity())

W = Linear(
    input_dim=3, output_dim=3, weights_init=initialization.Identity(2),
    biases_init=initialization.Constant(0))

H.initialize()
W.initialize()

out = H.apply(W.apply(x))



f = theano.function([x], out)
# Procedures

# Main
