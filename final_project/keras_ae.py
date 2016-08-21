#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
import numpy as np
import pandas as pd

def test():
    np.random.choice(vocab_size,1,p=dist)[0]


from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import SimpleRNN, LSTM, RepeatVector

# Main
def main():
    print "Load Data"
    Xtr, char_dict = loadData()
    sx, sy, sz = Xtr.shape
    indices = np.random.permutation(sx)
    split = int(sx * 0.8)
    
    Xval = Xtr[split:,:,:]
    Xtr  = Xtr[:split,:,:]
    
    
    HIDDEN = 10
    EPOCHS  = 100
    BATCH_SIZE = 2048
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
    encoder     = Model(inputs,encoded)
    print "Compile"
    autoencoder.compile(optimizer="adadelta", loss="categorical_crossentropy")
    print "Train"
    autoencoder.fit(Xtr, Xtr,
                    shuffle=True,
                    nb_epoch=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(Xval, Xval))

    autoencoder.save("ae_1_lstm.h5")
    
    
if __name__ == "__main__":
    main()
