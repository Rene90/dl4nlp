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
                               default="train",
                               dest="mode")
args = parser.parse_args()

import numpy as np
import theano
from theano import tensor

from blocks.model import Model

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.extensions import FinishAfter, Printing, ProgressBar
#from blocks.extensions.saveload import load
from blocks.serialization import load

from dataset import Corpus, createDataset

if args.mode == "train":
    seq_len = 100
    dim = 10
    feedback_dim = 8

    # Build the bricks and initialize them
    transition = GatedRecurrent(name="transition", dim=dim,
                                activation=Tanh())
    generator = SequenceGenerator(
        Readout(readout_dim=num_states, source_names=["states"],
                emitter=SoftmaxEmitter(name="emitter"),
                feedback_brick=LookupFeedback(
                    num_states, feedback_dim, name='feedback'),
                name="readout"),
        transition,
        weights_init = IsotropicGaussian(0.01),
        biases_init  = Constant(0),
        name = "generator"
    )
    generator.push_initialization_config()
    transition.weights_init = Orthogonal()
    generator.initialize()

    # Build the cost computation graph.
    x = tensor.lmatrix('data')
    cost = aggregation.mean(generator.cost_matrix(x[:, :]).sum(),
                            x.shape[1])
    cost.name = "sequence_log_likelihood"

    algorithm = GradientDescent(
        cost=cost,
        parameters=list(Selector(generator).get_parameters().values()),
        step_rule=Adam())
    main_loop = MainLoop(
        algorithm=algorithm,
        data_stream=DataStream(
            MarkovChainDataset(rng, seq_len),
            iteration_scheme=ConstantScheme(batch_size)),
        model=Model(cost),
        extensions=[FinishAfter(after_n_batches=num_batches),
                    TrainingDataMonitoring([cost], prefix="this_step",
                                           after_batch=True),
                    TrainingDataMonitoring([cost], prefix="average",
                                           every_n_batches=100),
                    Checkpoint(save_path, every_n_batches=500),
                    Printing(every_n_batches=100)])
    main_loop.run()
elif mode == "sample":
    main_loop = load(open(save_path, "rb"))
    generator = main_loop.model.get_top_bricks()[-1]

    sample = ComputationGraph(generator.generate(
        n_steps=steps, batch_size=1, iterate=True)).get_theano_function()

    states, outputs, costs = [data[:, 0] for data in sample()]

    numpy.set_printoptions(precision=3, suppress=True)
    print("Generation cost:\n{}".format(costs.sum()))

    freqs = numpy.bincount(outputs).astype(floatX)
    freqs /= freqs.sum()
    print("Frequencies:\n {} vs {}".format(freqs,
                                           MarkovChainDataset.equilibrium))

    trans_freqs = numpy.zeros((num_states, num_states), dtype=floatX)
    for a, b in zip(outputs, outputs[1:]):
        trans_freqs[a, b] += 1
    trans_freqs /= trans_freqs.sum(axis=1)[:, None]
    print("Transition frequencies:\n{}\nvs\n{}".format(
        trans_freqs, MarkovChainDataset.trans_prob))
else:
    assert False
