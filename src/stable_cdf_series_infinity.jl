# Asymptotic tail series for the standard stable cdf in the (M) parameterization:
#
#   1 - F(x) ≈ (1/π) Σₖ (-1)ᵏ [Γ(α(k+1))/(k+1)!] (1+ζ²)^((k+1)/2)
#                    sin((k+1)(πα/2 - atan ζ)) (x-ζ)^(-α(k+1))
#
# valid for x above the tail_series_threshold bound. stable_ccdf_series_infinity
# returns the upper-tail probability itself, at full relative accuracy — use it
# directly for tail probabilities instead of 1 - cdf, which cancels.
function stable_ccdf_series_infinity(x::Float64, a::Float64, b::Float64,
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
    return val / pi
end

stable_cdf_series_infinity(x::Float64, a::Float64, b::Float64, max_coef::Integer) =
    1 - stable_ccdf_series_infinity(x, a, b, max_coef)
