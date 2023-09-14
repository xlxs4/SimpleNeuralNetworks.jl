relu(x) = ifelse(x < 0, zero(x), x) # Faster than max(zero(x), x), NaN-preserving
