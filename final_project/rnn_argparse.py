#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", action="store",
                               default="./ae_rnn.h5",
                               dest="model")

parser.add_argument("--hidden", action="store",
                               default=10,
                               type=int,
                               dest="hidden_size")


parser.add_argument("--epochs", action="store",
                               default=100,
                               type=int,
                               dest="epoch_size")


parser.add_argument("--batch", action="store",
                               default=1024,
                               type=int,
                               dest="batch_size")

parser.add_argument("--mode", action="store",
                               default="train",
                               dest="mode")

parser.add_argument("--input", action="store",
                               default="brown_words.npy",
                               dest="input")

parser.add_argument("--output", action="store",
                               default="brown_words.npy",
                               dest="output")

parser.add_argument("--dict", action="store",
                               default="char_dict.npy",
                               dest="dict")

#parser.add_argument("--sample_size", action="store",
#                               default=500,
#                               type=int,
#                               dest="sample_size")

def getArguments():
    return parser.parse_args()
