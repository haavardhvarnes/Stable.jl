# Cdf of a standard (μ = 0, σ = 1) stable law in Zolotarev's (M) parameterization
# with stability index `a` and skewness `b`.
stable_cdf(x::Real, a::Real, b::Real) = stable_cdf(Float64(x), Float64(a), Float64(b))

function stable_cdf(x::Float64, a::Float64, b::Float64)
    b == 0.0 && return stable_sym_cdf(x, a)

    z = -b * tan(pi * a / 2)
    # reflection identity: F(x; α, β) = 1 - F(-x; α, -β)
    x < z && return 1.0 - stable_cdf(-x, a, -b)

    a >= 1.1 || throw(DomainError(a,
        "asymmetric (β ≠ 0) stable cdf requires α ∈ [1.1, 2]"))
    n_inf = 80

    # smallest x for which the truncated series at infinity reaches machine precision
    min_inf_x = ((1 + z^2)^(n_inf / 2) * a / (pi * eps(Float64)) *
                 gamma(a * n_inf) / gamma(n_inf))^(1 / (a * n_inf - 1)) + z

    return x > min_inf_x ? stable_cdf_series_infinity(x, a, b, n_inf) :
           stable_cdf_integral(x, a, b)
end
