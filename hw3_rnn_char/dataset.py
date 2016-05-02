#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import numpy as np
import pandas as pd

def createH5Dataset(path,sequence):
    f = h5py.File(path, mode='w')

    length = len(sequence)-1

    minibatches = 5
    seq_len = 5


    #char_in  = f.create_dataset('inchar', (, 117), dtype='uint8')
    char_out = f.create_dataset('outchar', (8123, 2), dtype='uint8')

    features[...] = X
    targets[...] = y

    f.flush()
    f.close()

def tokenize_corpus(txt):
    txt = txt.replace("\ \ ", " ")
    txt = list(txt.lower())
    return txt

if __name__ == "__main__":
    tokens = []
    with open("corpus.txt") as fh:
        tokens = tokenize_corpus(fh.read())

    words,vocab = pd.factorize(tokens)
    print words, vocab
    #createH5Dataset("dataset_rnn.hdf5", X, y)

#train_data = H5PYDataset('dataset.hdf5', which_sets=('train',), load_in_memory=True)
#test_data  = H5PYDataset('dataset.hdf5', which_sets=('test',), load_in_memory=True)
