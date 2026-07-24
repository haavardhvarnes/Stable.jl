# Series expansion of the standard stable pdf around the origin (x near ζ).
# Currently not wired into the `stable_pdf` dispatcher; kept as a reference
# implementation from qastable.
function stable_pdf_series_zero(x::Float64, a::Float64, b::Float64,
                                max_coef::Integer)
    zeta = -b * tan(pi * a / 2)
    angle = pi / 2 + atan(zeta) / a
    inv_geometric = (1 + zeta^2)^(-1 / (2a))
    x_shift = x - zeta

    val = 0.0
    geometric_part = 1.0
    x_part = 1.0
    for k in 0:max_coef
        geometric_part *= inv_geometric
        # Γ((k+1)/α) / k! via loggamma to avoid overflow for large k
        gamma_part = exp(loggamma((k + 1) / a) - loggamma(k + 1.0))
        val += gamma_part * geometric_part * sin(angle * (k + 1)) * x_part
        x_part *= x_shift
    end
    return val / (a * pi)
end
