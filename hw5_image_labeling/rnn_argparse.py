#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--retrain", action="store_true",
                                 default=False,
                                 dest="retrain")

parser.add_argument("--model", action="store",
                               default="./model_checkpoints/savepoint.pkl",
                               dest="model")

parser.add_argument("--hidden", action="store",
                               default=200,
                               type=int,
                               dest="hidden")

parser.add_argument("--mode", action="store",
                               default="train",
                               dest="mode")

parser.add_argument("--corpus", action="store",
                               default="corpus.txt",
                               dest="corpus")

parser.add_argument("--sample_size", action="store",
                               default=500,
                               type=int,
                               dest="sample_size")

def getArguments():
    return parser.parse_args()
