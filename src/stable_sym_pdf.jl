# Pdf of a standard symmetric (β = 0) stable law in the (M) parameterization.
function stable_sym_pdf(x::Float64, a::Float64)
    if a == 2.0
        # α = 2 is Gaussian with cf exp(-t²), i.e. Normal(0, √2)
        return exp(-x^2 / 4) / (2 * sqrt(pi))
    elseif a == 1.0
        # Cauchy
        return 1 / (pi * (1 + x^2))
    elseif 0.5 <= a < 2.0
        n_inf = 42
        # smallest |x| for which the truncated series at infinity reaches machine precision
        min_inf_x = (a / (pi * eps(Float64)) *
                     gamma(a * n_inf) / gamma(n_inf))^(1 / (a * n_inf - 1))
        xa = abs(x)
        return xa > min_inf_x ? stable_pdf_series_infinity(xa, a, 0.0, n_inf) :
               stable_sym_pdf_fourier_integral(x, a)
    else
        throw(DomainError(a, "symmetric stable pdf requires α ∈ [0.5, 2]"))
    end
end
