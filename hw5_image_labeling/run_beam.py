#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#
generator = SequenceGenerator(
    Readout(readout_dim = vocab_size,
            source_names = ["states"], # transition.apply.states ???
            emitter = SoftmaxEmitter(name="emitter"),
            feedback_brick = LookupFeedback(
                vocab_size,
                input_dim,
                name = 'feedback'
            ),
            name = "readout"
    ),
    SimpleRecurrent(
        name = "transition",
        activation = Tanh(),
        dim = 512
    ),
    weights_init = IsotropicGaussian(0.01),
    biases_init  = Constant(0),
    name = "generator"
)
generator.push_initialization_config()
generator.transition.weights_init = IsotropicGaussian(0.01)
generator.initialize()

main_loop = load(open(args.model, "rb"))
old_parameters = main_loop.model.get_parameters()
old_parameters["/generator/with_fake_attention/transition.initial_state"] = image[...]

sequence = tensor.lmatrix("sequence")
generated = generator.generate(n_steps=sequence.shape[0], batch_size=sequence.shape[1])
model = Model(generated)
model.set_parameter_values(old_parameters)
label = VariableFilter(bricks=[generator], name="outputs")(model)[1]

from blocks.blocks.search import BeamSearch
beamSearch = BeamSearch(label)
beamSearch.search(




