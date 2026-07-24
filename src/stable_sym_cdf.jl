# Cdf of a standard symmetric (β = 0) stable law in the (M) parameterization.
function stable_sym_cdf(x::Float64, a::Float64)
    if a == 2.0
        # α = 2 is Gaussian with cf exp(-t²), i.e. Normal(0, √2)
        return (1 + erf(x / 2)) / 2
    elseif a == 1.0
        # Cauchy
        return atan(x) / pi + 0.5
    elseif 0.5 <= a < 2.0
        # same tail bound as for the pdf series; the cdf series bound is strictly smaller
        n_inf = 42
        xa = abs(x)
        if xa > tail_series_threshold(a, 0.0, n_inf)
            # the series sum is the tail probability itself; return it directly
            # in the lower tail so the tiny value keeps full relative accuracy
            return x > 0 ? 1.0 - stable_ccdf_series_infinity(xa, a, 0.0, n_inf) :
                   stable_ccdf_series_infinity(xa, a, 0.0, n_inf)
        else
            return stable_sym_cdf_integral(x, a)
        end
    else
        throw(DomainError(a, "symmetric stable cdf requires α ∈ [0.5, 2]"))
    end
end
