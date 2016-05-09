#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import theano
import numpy as np

def sample_chars(model, num_chars, vocab_size, init_char=0):

    def get_var_from(name,vars):
        return vars[map(lambda x: x.name, vars).index(name)]

    v_inchar  = get_var_from("inchar",model.variables)
    v_softmax = get_var_from("softmax_apply_output",model.variables)
    v_init    = get_var_from("initial_state",model.shared_variables)
    v_states  = get_var_from("H_apply_states",model.intermediary_variables)

    f = theano.function([v_inchar], v_softmax, updates=[(v_init, v_states[0][0])])
    #f = theano.function([v_inchar], v_softmax)

    seq = np.array([[init_char]], dtype=np.int32)
    out = seq
    for i in range(num_chars):
        dist = f(out)[0]
        sample = np.random.choice(vocab_size,1,p=dist)
        seq = np.hstack([seq, np.atleast_2d(sample)])

    return seq

def sample_text(model, num_chars, corpus):
     return "".join(corpus.decode(sample_chars(model, num_chars, corpus.vocab_size())[0]))
