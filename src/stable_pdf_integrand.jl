# Integrand of Nolan's stationary-phase integral representation of the
# standard stable pdf. Reference implementation (see also stdstable_pdf_quad
# in stable_pdf_fft.jl, which integrates it adaptively).
function stable_pdf_integrand(theta::Float64, x::Float64, alpha::Float64,
                              beta::Float64)
    t = theta
    a = alpha
    b = beta

    if a != 1.0
        z = -b * tan(pi * a / 2)
        x == z && return 0.0

        # update input according to the reflection symmetry
        if x < z
            x = -x
            b = -b
            z = -b * tan(pi * a / 2)
        end
        t_0 = atan(-z) / a
        cos_t = cos(t)
        exp_1 = 1 / (a - 1)
        exp_2 = a * exp_1

        V = cos(a * t_0)^exp_1 * (cos_t / sin(a * (t + t_0)))^exp_2 *
            cos(a * t_0 + (a - 1) * t) / cos_t
        g = (x - z)^exp_2 * V
        I = g * exp(-g)
    else
        t_0 = pi / 2
        V = (2 / pi) * ((pi / 2) + b * t) / cos(t) *
            exp((1 / b) * (pi / 2 + b * t) * tan(t))
        g_a1 = exp(-(pi / 2) * (x / b)) * V
        I = g_a1 * exp(-g_a1)
    end

    # zero outside the domain of the integrand
    if t <= -t_0 + eps(Float64) || t >= pi / 2 - eps(Float64)
        I = 0.0
    end
    return isnan(I) ? 0.0 : I
end
