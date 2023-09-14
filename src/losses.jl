function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

epseltype(x) = eps(float(eltype(x)))

function crossentropy(ŷ, y; dims = 1, agg = mean, eps::Real = epseltype(ŷ))
    return agg(.-sum(xlogy.(y, ŷ .+ eps); dims = dims))
end
