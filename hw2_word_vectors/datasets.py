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
        super(WordWindow, self).__init__(**kwargs)

    def get_data(self, state=None, request=None):
        data = [((self.vocabulary[0][i],self.vocabulary[0][i+2]),self.vocabulary[0][i+1]) for i in xrange(len(self.corpus)-2)]
        x, y = zip(*data)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def open(self):
        return open('./data/small')

    def close(self,fh):
        fh.close()
