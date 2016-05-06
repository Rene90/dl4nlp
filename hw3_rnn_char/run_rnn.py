#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import numpy
import theano
from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Linear, Tanh, Rectifier, NDimensionalSoftmax
from blocks.bricks.lookup import LookupTable
from blocks.bricks.cost import CategoricalCrossEntropy

from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT

from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint

from dataset import createDataset
train_data,vocab_size = createDataset()

def initLayers(layers):
    for l in layers: l.initialize()

SAVE_PATH = "./model_checkpoints/"
HIDDEN_DIM = 100
VOCAB_DIM = vocab_size
#print "Vocabulary size of", vocab_size

# input
x = tensor.imatrix('inchar')
y = tensor.imatrix('outchar')

#
W = LookupTable(
    name = "W1",
    dim = HIDDEN_DIM,
    length = VOCAB_DIM,
    weights_init = initialization.IsotropicGaussian(0.01),
    biases_init = initialization.Constant(0)
)
# recurrent history weight
H = SimpleRecurrent(
    name = "H",
    dim = HIDDEN_DIM,
    activation = Rectifier(),
    weights_init = initialization.IsotropicGaussian(0.01)
)
#
S = Linear(
    name = "W2",
    input_dim = HIDDEN_DIM,
    output_dim = VOCAB_DIM,
    weights_init = initialization.IsotropicGaussian(0.01),
    biases_init = initialization.Constant(0)
)

A = NDimensionalSoftmax(
    name = "softmax"
)

initLayers([W,H,S])
activations = W.apply(x)
hiddens = H.apply(activations)
activations2 = S.apply(hiddens)
y_hat = A.apply(activations2, extra_ndim=1)
cost = A.categorical_cross_entropy(y, activations2, extra_ndim=1).mean()

cg = ComputationGraph(cost)
#print VariableFilter(roles=[WEIGHT])(cg.variables)
W1,H,W2 = VariableFilter(roles=[WEIGHT])(cg.variables)

main_loop = MainLoop(
    data_stream = DataStream(
        train_data,
        iteration_scheme = SequentialScheme(
            train_data.num_examples,
            batch_size = 1
        )
    ),
    algorithm = GradientDescent(
        cost  = cost,
        parameters = cg.parameters,
        step_rule = Scale(learning_rate=0.1)
    ),
    extensions = [
        #DataStreamMonitoring(variables=[cost]),
        FinishAfter(after_n_epochs=10),
        Printing(),
        TrainingDataMonitoring([cost,], after_batch=True),
        #Checkpoint(SAVE_PATH, every_n_batches=2000),
    ],
    model = Model(cost)
)
main_loop.run()

#import blocks.serialization.load

model = main_loop.model
#print dir(model.variables)
#print model.shared_variables


def sample_chars(model,num_chars):
    v_inchar = model.variables[map(lambda x: x.name, model.variables).index("inchar")]
    v_softmax = model.variables[map(lambda x: x.name, model.variables).index("softmax_log_probabilities_output")]
    f = theano.function([v_inchar], v_softmax)
    seq = np.array([[0]], dtype=np.int32)
    for i in range(num_chars):
        out = f(seq.astype(np.int32)).argmax(1)
        seq = np.hstack([seq, np.atleast_2d(out)])
