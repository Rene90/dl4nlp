#!/usr/bin/python
#
# author: koller
#
# description: just copied the template from the website
#
from six import wraps

from theano import tensor

from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.bricks.interfaces import Initializable
from blocks.bricks.base import Application, application, Brick, lazy
from blocks.roles import add_role, WEIGHT

from blocks.utils import (pack, shared_floatx_nans, shared_floatx_zeros,
                          dict_union, dict_subset, is_shared_variable)

class MySimpleRecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        super(MySimpleRecurrent, self).__init__(**kwargs)
        self.dim = dim
        self.children = [activation]

    @property
    def W(self):
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (MySimpleRecurrent.apply.sequences +
                    MySimpleRecurrent.apply.states):
            return self.dim
        return super(MySimpleRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim), name="W"))
        add_role(self.parameters[0], WEIGHT)

        # NB no parameters for initial state

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=['context'])
    def apply(self, inputs, states, mask=None, **kwargs):
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(contexts=["context"])
    def initial_states(self, batch_size, *args, **kwargs):
        init = kwargs["context"]
        return init.T

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.apply.states
