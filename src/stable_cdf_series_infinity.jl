# Asymptotic tail series for the standard stable cdf in the (M) parameterization:
#
#   F(x) ≈ 1 - (1/π) Σₖ (-1)ᵏ [Γ(α(k+1))/(k+1)!] (1+ζ²)^((k+1)/2)
#                     sin((k+1)(πα/2 - atan ζ)) (x-ζ)^(-α(k+1))
#
# valid for x above the `min_inf_x` bound computed by the dispatchers.
function stable_cdf_series_infinity(x::Float64, a::Float64, b::Float64,
                                    max_coef::Integer)
    zeta = -b * tan(pi * a / 2)
    angle = pi / 2 * a - atan(zeta)
    sqrt_1_plus_zeta2 = sqrt(1 + zeta^2)
    x_to_minus_a = (x - zeta)^(-a)

    val = 0.0
    term_sign = 1.0
    geometric_part = 1.0
    x_part = 1.0
    for k in 0:max_coef
        geometric_part *= sqrt_1_plus_zeta2
        x_part *= x_to_minus_a
        # Γ(α(k+1)) / (k+1)! via loggamma to avoid overflow for large k
        gamma_part = exp(loggamma(a * (k + 1)) - loggamma(k + 2.0))
        val += term_sign * gamma_part * geometric_part * sin(angle * (k + 1)) * x_part
        term_sign = -term_sign
    end
    return 1 - val / pi
end
