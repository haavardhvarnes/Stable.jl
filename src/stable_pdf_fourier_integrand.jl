# Integrand of the Fourier representation of the standard stable pdf.
# Reference implementation; the hard-coded quadrature in
# stable_pdf_fourier_integral.jl evaluates this expression inline.
function stable_pdf_fourier_integrand(t::Float64, x::Float64, a::Float64, b::Float64)
    if a == 1.0
        h = x * t + (2 / pi) * b * t * log(t)
    else
        z = -b * tan(pi * a / 2)
        h = (x - z) * t + z * t^a
    end
    return cos(h) * exp(-t^a)
end
