#!/usr/bin/python
# coding: utf-8
#
# author:
#
# date:
# description:
#
import numpy as np
import pandas as pd

from nltk.corpus import brown

from itertools import izip

from fuel.datasets.base import Dataset

class ToyCorpus(Dataset):
    def __init__(self, **kwargs):
        self.provides_sources = ('features', 'targets')
         # for technical reasons
        self.axis_labels = None
        with self.open() as fh:
            self.corpus = fh.read().split()
        #print self.corpus
        self.vocabulary_size = len(set(self.corpus))
        self.vocabulary = pd.factorize(self.corpus)
        self.num_instances = len([((self.vocabulary[0][i],self.vocabulary[0][i+2]),self.vocabulary[0][i+1]) for i in xrange(len(self.corpus)-2)])
        super(ToyCorpus, self).__init__(**kwargs)

    def get_data(self, state=None, request=None):
        data = [((self.vocabulary[0][i],self.vocabulary[0][i+2]),self.vocabulary[0][i+1]) for i in xrange(len(self.corpus)-2)]
        x, y = zip(*data)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def open(self):
        return open('./data/small')

    def close(self,fh):
        fh.close()

class BrownCorpus(Dataset):
    def __init__(self, window_size=1, **kwargs):
        self.provides_sources = ('context', 'center')
         # for technical reasons
        self.axis_labels = None
        self.window_size = window_size
        res = pd.factorize(brown.words())
        self.corpus = res[0]
        self.word_dict = res[1]
        self.vocabulary_size = len(self.word_dict)
        self.num_instances = len(self.corpus) - (2 * window_size)
        super(BrownCorpus, self).__init__(**kwargs)

    def next_window(self):
        step = 0
        indices = range(2*self.window_size+1)
        del indices[self.window_size]
        indices = np.array(indices)
        for i in xrange(self.num_instances):
            yield self.corpus[indices+i], self.corpus[i+self.window_size]

    def get_data(self, state=None, request=None):
        x, y = izip(*list(self.next_window()))
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

if __name__ == "__main__":
    bc = BrownCorpus(window_size=1)
    brown_data = bc.get_data()
    np.save("brown_corpus_context.npy", brown_data[0])
    np.save("brown_corpus_center.npy", brown_data[1])
