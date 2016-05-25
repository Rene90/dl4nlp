#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
class MySimpleRecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        self.dim = dim
        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(MySimpleRecurrent, self).__init__(**kwargs)

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
