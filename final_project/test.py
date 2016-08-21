#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
from keras.models import load_model

from rnn_argparse import getArguments
from dataset import loadData, decode, encode

import numpy as np

# Main
def main():
    args = getArguments()
    SPLIT = 100

    #
    # prepare DATA
    #    
    print "Load Data"
    X, char_dict = loadData()
    X = X[:100]

    print "load existing model: ", args.model
    autoencoder = load_model(args.model)

    y_hat = autoencoder.predict(X)

    for idx in range(X.shape[0]):
        x1 = decode(np.argmax(X[idx],    axis=1),char_dict)
        x2 = decode(np.argmax(y_hat[idx],axis=1),char_dict)
        print "".join(x1), "".join(x2)
        raw_input()
    

if __name__ == "__main__":
    main()
