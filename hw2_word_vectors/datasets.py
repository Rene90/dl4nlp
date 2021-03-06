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

from itertools import izip
from fuel.datasets.base import Dataset

class ToyCorpus(Dataset):
    def __init__(self, **kwargs):
        self.provides_sources = ('context', 'center')
         # for technical reasons
        self.axis_labels = None
        with self.open() as fh:
            self.corpus = fh.read().split()
        #print self.corpus
        self.vocabulary_size = len(set(self.corpus))
        res = pd.factorize(self.corpus)
        self.corpus = res[0]
        self.word_dict = res[1]

        self.num_instances = len([((self.corpus[i],self.corpus[i+2]),self.corpus[i+1]) for i in xrange(len(self.corpus)-2)])
        super(ToyCorpus, self).__init__(**kwargs)

    def get_data(self, state=None, request=None):
        data = [((self.corpus[i],self.corpus[i+2]),self.corpus[i+1]) for i in xrange(len(self.corpus)-2)]
        x, y = zip(*data)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def open(self):
        return open('./data/small')

    def close(self,fh):
        fh.close()

class BrownCorpus(Dataset):
    def __init__(self, window_size=1, load=False, **kwargs):
        self.provides_sources = ('context', 'center')
         # for technical reasons
        self.axis_labels = None
        self.load = load
        self.window_size = window_size
        res = (np.load("brown_corpus.npy"),np.load("brown_dict.npy"))
        self.corpus = res[0]
        self.word_dict = res[1]
        self.vocabulary_size = len(self.word_dict)
        self.num_instances = len(self.corpus) - (2 * window_size)
        self.x, self.y = self.prepareStream()
        super(BrownCorpus, self).__init__(**kwargs)

    def next_window(self):
        indices = range(2*self.window_size+1)
        del indices[self.window_size]
        indices = np.array(indices)
        for i in xrange(self.num_instances):
            yield self.corpus[indices+i], self.corpus[i+self.window_size]

    def prepareStream(self):
        if self.load:
            x, y = np.load("brown_corpus_context.npy"), np.load("brown_corpus_center.npy")
        else:
            x, y = [], []
            for xx,yy in self.next_window():
                x.append(xx); y.append(yy)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def get_data(self, state=None, request=None):
        if not request:
            request = range(self.num_instances)
        return self.x[request], self.y[request]

def factorized_brown_corpus():
    from nltk.corpus import brown
    res = pd.factorize(brown.words())
    np.save("brown_corpus.npy", res[0])
    np.save("brown_dict.npy", res[1])

if __name__ == "__main__":
    print "factorize brown corpus"
    factorized_brown_corpus()
    print "load corpus"
    bc = BrownCorpus(window_size=1)
    brown_data = bc.get_data()
    print brown_data
    print "save it"
    np.save("brown_corpus_context.npy", brown_data[0])
    np.save("brown_corpus_center.npy", brown_data[1])
