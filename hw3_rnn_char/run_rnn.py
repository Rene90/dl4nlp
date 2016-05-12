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
parser.add_argument("--mode", action="store",
                               default="rnn",
                               dest="mode")
parser.add_argument("--hidden", action="store",
                               default=200,
                               type=int,
                               dest="hidden")
parser.add_argument("--corpus", action="store",
                               default="corpus.txt",
                               dest="corpus")

import numpy
import numpy as np
import theano
from theano import tensor

from blocks.model import Model

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint, load

from dataset import Corpus, createDataset
from rnn_model import create_rnn


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.retrain:
        main_loop = load(args.model)
    else:
        # create Corpus and Dateset
        corpus = Corpus(open(args.corpus).read())
        train_data,vocab_size = createDataset(
            corpus=corpus,
            sequence_length=750,
            repeat=20
        )
        # create Computation Graph
        cg, layers, y_hat, cost = create_rnn(args.hidden, vocab_size, mode=args.mode)
        # create training loop
        main_loop = MainLoop(
            data_stream = DataStream(
                train_data,
                iteration_scheme = SequentialScheme(
                    train_data.num_examples,
                    batch_size = 50
                )
            ),
            algorithm = GradientDescent(
                cost  = cost,
                parameters = cg.parameters,
                step_rule = Adam()
            ),
            extensions = [
                #DataStreamMonitoring(variables=[cost]),
                FinishAfter(),
                Printing(),
                ProgressBar(),
                TrainingDataMonitoring([cost,], after_batch=True),
                Checkpoint(args.model, every_n_epochs=1),
            ],
            model = Model(y_hat)
        )
    
    main_loop.run()

    
