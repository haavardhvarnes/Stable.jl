# Pdf of a standard (μ = 0, σ = 1) stable law in Zolotarev's (M) parameterization
# with stability index `a` and skewness `b`. Selects between the symmetric
# routines, the tail series and the Fourier integral (see qastable / Ament & O'Neil).
stable_pdf(x::Real, a::Real, b::Real) = stable_pdf(Float64(x), Float64(a), Float64(b))

function stable_pdf(x::Float64, a::Float64, b::Float64)
    b == 0.0 && return stable_sym_pdf(x, a)

    z = -b * tan(pi * a / 2)
    # reflection identity: f(x; α, β) = f(-x; α, -β)
    x < z && return stable_pdf(-x, a, -b)

    # number of terms used in the tail series expansion
    if a >= 1.1
        n_inf = 80
    elseif 0.5 <= a <= 0.9
        n_inf = 90
    else
        throw(DomainError(a,
            "asymmetric (β ≠ 0) stable pdf requires α ∈ [0.5, 0.9] ∪ [1.1, 2]"))
    end

    # smallest x for which the truncated series at infinity reaches machine precision
    min_inf_x = ((1 + z^2)^(n_inf / 2) * a / (pi * eps(Float64)) *
                 gamma(a * n_inf) / gamma(n_inf))^(1 / (a * n_inf - 1)) + z

    return x > min_inf_x ? stable_pdf_series_infinity(x, a, b, n_inf) :
           stable_pdf_fourier_integral(x, a, b)
end
