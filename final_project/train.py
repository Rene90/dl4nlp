#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.layers import SimpleRNN, LSTM, RepeatVector
from keras.callbacks import ModelCheckpoint, EarlyStopping

from rnn_argparse import getArguments
from dataset import loadData

import numpy as np

# Main
def main():
    args = getArguments()
    HIDDEN = args.hidden_size
    EPOCHS  = args.epoch_size
    BATCH_SIZE = args.batch_size

    #
    # prepare DATA
    #    
    print "Load Data"
    X_tr = np.load(args.input)
    y_tr = np.load(args.output) if args.input != args.output else X_tr
    
    sx, sy, sz = X_tr.shape
    split = int(sx * 0.8)
    
    X_val = X_tr[split:,:,:]
    X_tr  = X_tr[:split,:,:]
    
    y_val = y_tr[split:,:,:]
    y_tr  = y_tr[:split,:,:]

    
    if args.mode == "train":
        print "Define RNN"
        inputs  = Input(shape=(sy,sz))
        encoded = LSTM(HIDDEN,
                       activation="relu",
                       init="normal")(inputs)
        
        decoded = RepeatVector(sy)(encoded)
        decoded = LSTM(sz,
                       return_sequences=True,
                       activation="softmax",
                       init="normal")(decoded)

        autoencoder = Model(inputs,decoded)
    elif args.mode == "retrain":
        print "load existing model: ", args.model
        autoencoder = load_model(args.model)
    print "Compile"
    autoencoder.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    print "Train"
    autoencoder.fit(X_tr, y_tr,
                    shuffle=True,
                    nb_epoch=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    callbacks=[
                        ModelCheckpoint(args.model,save_best_only=True),
                        #EarlyStopping(patience=20)
                    ]
    )

    #autoencoder.save(args.model)

if __name__ == "__main__":
    main()
