#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#


# Procedures

# Main
import numpy as np
import pandas as pd

from nltk.corpus import brown


def encode(dec,chars):
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    return map(lambda x: char_to_ix[x], dec)
def decode(enc,chars):
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    return map(lambda x: ix_to_char[x], enc)

def makeOneHotTensor(words,chars,reverse=False):
    sx = len(words)
    sy = max(map(len,words))
    sz = len(chars)

    one_hot = np.zeros((sx,sy,sz), dtype=np.bool_)

    for i in range(sx):
        word = words[i] if not reverse else words[i][::-1]
        # padding
        word = word+"_"*(sy-len(word))
        # encoding
        vals = encode(word,chars)
        one_hot[i,range(len(vals)),vals] = 1

    return one_hot

def saveData():
    # i just wanna have character based words
    words_dataset = (filter(lambda x: ((3 < len(x) < 8) and x.isalpha()),
                            brown.words()))
    words_dataset = map(lambda x: x.lower(), words_dataset)

    chars = list(reduce(lambda x,y: x | set(y), words_dataset, set()))
    chars.append("_")
    data_size, vocab_size = len(words_dataset), len(chars)

    indices = np.random.permutation(len(words_dataset))

    one_hot = makeOneHotTensor(words_dataset,chars)[indices]
    np.save("brown_words.npy", one_hot)
    
    one_hot = makeOneHotTensor(words_dataset,chars,reverse=True)[indices]
    np.save("brown_words_rev.npy", one_hot)
    
    char_dict = np.array(decode(range(vocab_size),chars))
    np.save("char_dict.npy", char_dict)

    return one_hot, char_dict

def loadData():
    try:
        data      = np.load("brown_words.npy")
        char_dict = np.load("char_dict.npy")
    except:
        print "Couldn't open existing Data - generate new data"
        data, char_dict = saveData()
    return data, char_dict

if __name__ == "__main__":
    if raw_input("Do you want to (re)generate the dataset?(y/n)") == "y":
        saveData()
    print "Done"
