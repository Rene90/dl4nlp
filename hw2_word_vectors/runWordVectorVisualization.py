#!/usr/bin/python
#
# author:
#
# date:
# description:
#
from extensions import visualizeWordVector

import numpy as np

# Procedures
def main():
    for i in xrange(10,109,10):
        brown_dict = np.load("brown_dict.npy")
        W1 = np.load("./data/w1_{}.npy".format(i+1))
        #W2 = np.load("./data/w2_{}.npy".format(i))
        #word_vectors = (W1.get_value() + W2.get_value().T) / 2
        filename = "./data/wv_{}.png".format(i+1)
        print "processing vis{}...".format(i+1)
        visualizeWordVector(W1.T,brown_dict,filename)
        print "done"

# Main
if __name__ == "__main__":
    main()
