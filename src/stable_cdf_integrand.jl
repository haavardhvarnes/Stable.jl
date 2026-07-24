# Integrand of the Fourier representation of the standard stable cdf.
# Reference implementation; the hard-coded quadrature in stable_cdf_integral.jl
# evaluates this expression inline.
function stable_cdf_integrand(t::Float64, x::Float64, a::Float64, b::Float64)
    t_to_a = t^a
    z = -b * tan(a * pi / 2)
    h = (x - z) * t + z * t_to_a
    return sin(h) * exp(-t_to_a) / t
end
