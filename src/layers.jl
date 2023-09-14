struct Chain
    layers::Tuple
end

Chain(xs...) = Chain(xs)
function Chain(; kw...)
    isempty(kw) && return Chain(())
    return Chain(values(kw))
end

# https://github.com/FluxML/Zygote.jl/issues/1126#issuecomment-981191246
# init=x can disconnect the computational graph and make Zygote lose track of the grad
# https://github.com/FluxML/Flux.jl/pull/1809#issuecomment-1009516145
# afoldl needs an rrule
# https://github.com/EnzymeAD/Enzyme.jl/issues/805
# TODO: keep tabs on this
(c::Chain)(x) = foldl(|>, (x, c.layers...))

@concrete struct Dense
    activation::Any
    weight::Any
    bias::Any
end

function Dense(mapping::Pair{<:Int, <:Int}, activation = identity; kwargs...)
    return Dense(first(mapping), last(mapping), activation; kwargs...)
end

function Dense(in_dims::Int,
    out_dims::Int,
    activation = identity;
    init_weight = he_normal,
    init_bias = zeros32)
    weight = init_weight(out_dims, in_dims)
    bias = init_bias(out_dims)
    return Dense(activation, weight, bias)
end

# Don't broadcast activation if none was selected
@inline __apply_activation(::typeof(identity), x) = x
@inline __apply_activation(f, x) = f.(x)

@inline function (d::Dense)(x::AbstractVecOrMat)
    return __apply_activation(d.activation, d.weight * x .+ d.bias)
end
