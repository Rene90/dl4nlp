#!/usr/bin/python
# coding: utf-8
#
# author:
#
# date:
# description:
#
import numpy as np
from blocks.extensions import SimpleExtension

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#
# Store recent weight calculations with the given prefixes
# in numpy matrix format
#
class SaveWeights(SimpleExtension):
    def __init__(self, layers, prefixes, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(SaveWeights, self).__init__(**kwargs)
        self.step = 1
        self.layers = layers
        self.prefixes = prefixes

    def do(self, callback_name, *args):
        for i in xrange(len(self.layers)):
            filename = "%s_%d.npy" % (self.prefixes[i], self.step)
            np.save(filename, self.layers[i].get_value())
        self.step += 1
#
# Visualize word vectors in a 2-dimensional grid
# using PCA for dimensionality reduction
#
class VisualizeWordVectors(SimpleExtension):
    def __init__(self, layers, labels, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(VisualizeWordVectors, self).__init__(**kwargs)
        self.step = 1
        self.layers = layers
        self.labels = labels

    def do(self, callback_name, *args):
        word_vectors = (W1.get_value() + W2.get_value().T) / 2
        filename = "./npy_stored/wv_%d.png" % (self.step)
        visualizeWordVector(word_vectors,self.labels,filename)
        self.step += 1

def visualizeWordVector(word_vec,labels,path):
    #init pca on word vector
    pca = PCA(n_components=2)
    print word_vec.shape
    low_dim_embs = pca.fit_transform(word_vec)
    print low_dim_embs.shape
    # plot low dimensional word vectors into two-dimensional space
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(path)
