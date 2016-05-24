#!/usr/bin/python
#
# author:
#
# date:
# description:
#

import numpy as np
import theano
from theano import tensor

from rnn_argparse import getArguments

from blocks.model import Model

from blocks.initialization import IsotropicGaussian, Constant, Orthogonal
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from blocks.extensions.saveload import Checkpoint #, load

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.select import Selector

from blocks.graph import ComputationGraph

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.extensions import FinishAfter, Printing, ProgressBar
#from blocks.extensions.saveload import load
from blocks.serialization import load
from blocks.monitoring import aggregation # ???

from dataset import Corpus, createDataset

args = getArguments()

corpus = Corpus(open(args.corpus).read())
train_data,vocab_size = createDataset(
            corpus = corpus,
            sequence_length = 750,
            repeat = 20
        )

if args.mode == "train":
    seq_len = 100
    dim = 100
    feedback_dim = 100

    # Build the bricks and initialize them
    transition = GatedRecurrent(name="transition", dim=dim,
                                activation=Tanh())
    generator = SequenceGenerator(
        Readout(readout_dim = vocab_size,
                source_names = ["states"], # transition.apply.states ???
                emitter = SoftmaxEmitter(name = "emitter"),
                feedback_brick = LookupFeedback(
                    vocab_size,
                    feedback_dim,
                    name = 'feedback'
                ),
                name = "readout"),
        transition,
        weights_init = IsotropicGaussian(0.01),
        biases_init  = Constant(0),
        name = "generator"
    )
    generator.push_initialization_config()
    transition.weights_init = Orthogonal()
    generator.initialize()

    # Build the cost computation graph.
    x = tensor.lmatrix('inchar')

    cost = generator.cost(outputs=x)
    cost.name = "sequence_cost"

    algorithm = GradientDescent(
        cost = cost,
        parameters = list(Selector(generator).get_parameters().values()),
        step_rule = Adam(),
        # because we want use all the stuff in the training data
        on_unused_sources = 'ignore'
    )
    main_loop = MainLoop(
        algorithm=algorithm,
        data_stream = DataStream(
            train_data,
            iteration_scheme = SequentialScheme(
                train_data.num_examples,
                batch_size = 20
            )
        ),
        model=Model(cost),
        extensions=[
            FinishAfter(),
            TrainingDataMonitoring([cost], prefix="this_step",
                                           after_batch=True),
            TrainingDataMonitoring([cost], prefix="average",
                                           every_n_batches=100),
            Checkpoint(args.model, every_n_epochs=5),
            Printing(every_n_batches=100)])
    main_loop.run()

elif args.mode == "retrain":
    main_loop = load(open(args.model, "rb"))
    main_loop.run()

elif args.mode == "sample":
    main_loop = load(open(args.model, "rb"))
    # get the one and only brick in the computation graph
    generator = main_loop.model.get_top_bricks()[0]

    sample = ComputationGraph(generator.generate(
        n_steps = args.sample_size,
        batch_size = 1,
        iterate = True
    )).get_theano_function()

    states, outputs, costs = [data[:, 0] for data in sample()]

    print "".join(corpus.decode(outputs))
else:
    assert False
