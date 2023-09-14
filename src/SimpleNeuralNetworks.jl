module SimpleNeuralNetworks

using Random, Statistics
using ConcreteStructs
using Zygote

include("activations.jl")
include("losses.jl")
include("initializers.jl")
include("softmax.jl")
include("layers.jl")

export relu
export crossentropy
export he_normal, zeros32
export softmax
export Chain, Dense

end
