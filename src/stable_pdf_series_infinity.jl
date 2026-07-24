# Asymptotic tail series for the standard stable pdf in the (M) parameterization:
#
#   f(x) ≈ (α/π) Σₖ (-1)ᵏ [Γ(α(k+1))/k!] (1+ζ²)^((k+1)/2)
#                 sin((k+1)(πα/2 - atan ζ)) (x-ζ)^(-α(k+1)-1)
#
# valid for x above the `min_inf_x` bound computed by the dispatchers.
function stable_pdf_series_infinity(x::Float64, alpha::Float64, beta::Float64,
                                    max_coef::Integer)
    zeta = -beta * tan(pi * alpha / 2)
    angle = pi / 2 * alpha - atan(zeta)
    sqrt_1_plus_zeta2 = sqrt(1 + zeta^2)
    x_shift = x - zeta
    x_to_minus_a = x_shift^(-alpha)

    val = 0.0
    term_sign = 1.0
    geometric_part = 1.0
    x_part = 1 / x_shift
    for k in 0:max_coef
        geometric_part *= sqrt_1_plus_zeta2
        x_part *= x_to_minus_a
        # Γ(α(k+1)) / k! via loggamma to avoid overflow for large k
        gamma_part = exp(loggamma(alpha * (k + 1)) - loggamma(k + 1.0))
        val += term_sign * gamma_part * geometric_part * sin(angle * (k + 1)) * x_part
        term_sign = -term_sign
    end
    return val * alpha / pi
end
