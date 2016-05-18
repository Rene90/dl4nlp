
import theano
from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.bricks import Linear, Tanh, Rectifier, NDimensionalSoftmax
from blocks.bricks.lookup import LookupTable

from blocks.graph import ComputationGraph

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT

def initLayers(layers):
    for l in layers: l.initialize()

def create_rnn(hidden_dim, vocab_dim,mode="rnn"):
    # input
    x = tensor.imatrix('inchar')
    y = tensor.imatrix('outchar')

    # 
    W = LookupTable(
        name = "W1",
        #dim = hidden_dim*4,
        dim = hidden_dim,
        length = vocab_dim,
        weights_init = initialization.IsotropicGaussian(0.01),
        biases_init = initialization.Constant(0)
    )
    if mode == "lstm":
        # Long Short Term Memory
        H = LSTM(
            hidden_dim, 
            name = 'H',
            weights_init = initialization.IsotropicGaussian(0.01),
            biases_init = initialization.Constant(0.0)
        )
    else:
        # recurrent history weight
        H = SimpleRecurrent(
            name = "H",
            dim = hidden_dim,
            activation = Tanh(),
            weights_init = initialization.IsotropicGaussian(0.01)
        )
    # 
    S = Linear(
        name = "W2",
        input_dim = hidden_dim,
        output_dim = vocab_dim,
        weights_init = initialization.IsotropicGaussian(0.01),
        biases_init = initialization.Constant(0)
    )

    A = NDimensionalSoftmax(
        name = "softmax"
    )

    initLayers([W,H,S])
    activations = W.apply(x)
    hiddens = H.apply(activations)#[0]
    activations2 = S.apply(hiddens)
    y_hat = A.apply(activations2, extra_ndim=1)
    cost = A.categorical_cross_entropy(y, activations2, extra_ndim=1).mean()

    cg = ComputationGraph(cost)
    #print VariableFilter(roles=[WEIGHT])(cg.variables)
    #W1,H,W2 = VariableFilter(roles=[WEIGHT])(cg.variables)

    layers = (x, W, H, S, A, y)

    return  cg, layers, y_hat, cost
