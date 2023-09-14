function softmax(x; dims = 1)
    exps = exp.(x .- maximum(x; dims))
    return exps ./ sum(exps; dims)
end
