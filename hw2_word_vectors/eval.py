#!/usr/bin/python
# coding: utf-8
#
# author:
#
# date:
# description:
#
import numpy as np
from scipy.spacial.distance import cosine

EVAL_FILE = "data/word-test.v1.txt"

def load_samples():
    samples = []
    with open(EVAL_FILE) as fh:
        for idx,line in enumerate(fh.readlines()):
            if idx in (0,1): continue
            samples.append(line.split())
    return samples

def get_word_vector(word,wv,labels):
    rowIdx = np.where(labels == word)
    return wv[rowIdx,:]

if __name__ == "__main__":
    s = load_samples()
    wv = np.load("data/w1_102.npy").T
    labels = np.load("brown_dict.npy")
    print get_word_vector(s[0][0],wv,labels)
