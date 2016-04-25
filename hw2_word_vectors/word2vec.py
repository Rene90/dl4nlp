#!/usr/bin/python
# coding: utf-8
#
# author:
#
# date:
# description:
#
import numpy as np
import pandas as pd

import zipfile

from theano import tensor

import fuel
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.bricks import Linear, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.initialization import IsotropicGaussian, Constant

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph

from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, AdaGrad

from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks_extras.extensions.plot import Plot

from extensions import SaveWeights
from datasets import ToyCorpus

dataset = ToyCorpus()

# In[75]:

VOCAB_DIM = dataset.vocabulary_size
EMBEDDING_DIM = min(5,VOCAB_DIM)
CONTEXT = 1

def makeGraph():
    Xs = tensor.lmatrix("features")
    y = tensor.ivector('targets')

    w1 = LookupTable(name="w1", length=VOCAB_DIM, dim=EMBEDDING_DIM)
    w2 = Linear(name='w2', input_dim=EMBEDDING_DIM, output_dim=VOCAB_DIM)

    hidden = tensor.mean(w1.apply(Xs), axis=1)
    y_hat = Softmax().apply(w2.apply(hidden))

    w1.weights_init = w2.weights_init = IsotropicGaussian(0.01)
    w1.biases_init = w2.biases_init = Constant(0)
    w1.initialize()
    w2.initialize()

    cost = CategoricalCrossEntropy().apply(y, y_hat)

    cg = ComputationGraph(cost)
    W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)

    cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
    cost.name = "loss"



    return cg,(W1,W2),cost

#
# the actual training of the model
#
cg, (W1, W2), cost = makeGraph()
main = MainLoop(data_stream = DataStream(
                    dataset,
                    iteration_scheme=SequentialScheme(dataset.num_instances, batch_size=50)),
                algorithm = GradientDescent(
                    cost = cost,
                    parameters = cg.parameters,
                    step_rule = AdaGrad()),
                extensions = [
                    ProgressBar(),
                    FinishAfter(after_n_epochs=10),
                    #Printing(),
                    TrainingDataMonitoring(variables=[cost], after_batch=True),
                    SaveWeights(layers=[W1, W2], prefixes=["./w1","./w2"]),
])

main.run()
