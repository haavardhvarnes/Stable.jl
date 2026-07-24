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
        min_inf_x = (a / (pi * eps(Float64)) *
                     gamma(a * n_inf) / gamma(n_inf))^(1 / (a * n_inf - 1))
        xa = abs(x)
        if xa > min_inf_x
            upper_tail_cdf = stable_cdf_series_infinity(xa, a, 0.0, n_inf)
            return x > 0 ? upper_tail_cdf : 1.0 - upper_tail_cdf
        else
            return stable_sym_cdf_integral(x, a)
        end
    else
        throw(DomainError(a, "symmetric stable cdf requires α ∈ [0.5, 2]"))
    end
end
