# Cdf of a standard (μ = 0, σ = 1) stable law in Zolotarev's (M) parameterization
# with stability index `a` and skewness `b`.
stable_cdf(x::Real, a::Real, b::Real) = stable_cdf(Float64(x), Float64(a), Float64(b))

function stable_cdf(x::Float64, a::Float64, b::Float64)
    b == 0.0 && return stable_sym_cdf(x, a)

    a >= 1.1 || throw(DomainError(a,
        "asymmetric (β ≠ 0) stable cdf requires α ∈ [1.1, 2]"))
    z = -b * tan(pi * a / 2)
    n_inf = 80

    if x > tail_series_threshold(a, z, n_inf)
        return stable_cdf_series_infinity(x, a, b, n_inf)
    elseif x < -tail_series_threshold(a, -z, n_inf)
        # deep lower tail: F(x; β) = 1 - F(-x; -β), evaluated as the reflected
        # upper-tail series itself so the tiny probability keeps full relative
        # accuracy (1 - (1 - tiny) would cancel)
        return stable_ccdf_series_infinity(-x, a, -b, n_inf)
    else
        # the quadrature integrand realizes the reflection identity
        # analytically, so it is valid on both sides of ζ
        return stable_cdf_integral(x, a, b)
    end
end
