@inline _nfan() = 1, 1 # fan_in, fan_out
@inline _nfan(n) = 1, n # A vector is treated as a n×1 matrix
@inline _nfan(n_out, n_in) = n_in, n_out

function he_normal(dims::Integer...; kwargs...)
    he_normal(Xoshiro(1234), Float32, dims...; kwargs...)
end

function he_normal(rng::AbstractRNG,
    ::Type{T},
    dims::Integer...;
    gain::Real = √T(2)) where {T <: Real}
    std = gain / sqrt(T(first(_nfan(dims...))))
    return randn(rng, T, dims...) .* std
end

zeros32(dims...) = zeros(Float32, dims...)
