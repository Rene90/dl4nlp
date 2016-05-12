#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import theano
import numpy as np

from blocks.extensions.saveload import load
SAVE_PATH = "./model_checkpoints/lstm_h200_sl750.pkl"

from dataset import Corpus

def sample_chars(model, num_chars, vocab_size, init_char=0):

    def get_var_from(name,vars):
        return vars[map(lambda x: x.name, vars).index(name)]

    v_inchar  = get_var_from("inchar",model.variables)
    v_softmax = get_var_from("softmax_apply_output",model.variables)
    v_init    = get_var_from("initial_state",model.shared_variables)
    v_states  = get_var_from("lstm_apply_states",model.intermediary_variables)

    f = theano.function([v_inchar], v_softmax, updates=[(v_init, v_states[0][0])])
    #f = theano.function([v_inchar], v_softmax)

    seq = [init_char]
    for _ in xrange(num_chars):
        dist = f(np.atleast_2d(seq[-1]).astype(np.int32))[0]
        sample = np.random.choice(vocab_size,1,p=dist)[0]
        seq.append(sample)
    #print seq
    return seq

def sample_text(model, num_chars, corpus):
     return "".join(corpus.decode(sample_chars(model, num_chars, corpus.vocab_size())))

corpus = Corpus(open("corpus.txt").read())

main_loop = load(SAVE_PATH)
model = main_loop.model

print sample_text(model, 5000, corpus)
