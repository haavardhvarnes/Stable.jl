# FFT-based evaluation of the stable pdf on a vector of points, by numerical
# inversion of the characteristic function (Witkovsky; Carr & Madan;
# Borak, Härdle & Weron). Fast for large batches, e.g. inside likelihood
# evaluations — see logpdf_fft / negloglike.
#
# References:
# - https://www.researchgate.net/publication/281465412 (Witkovsky)
# - Madan et al (1999): http://faculty.baruch.cuny.edu/lwu/890/ADP_Transform.pdf
# - http://prac.im.pwr.edu.pl/~hugo/publ/SFB2005-008_Borak_Haerdle_Weron.pdf

# Characteristic function of the alpha-stable distribution in Nolan's
# 0-parameterization (= Zolotarev's (M) form), matching stable_pdf/stable_cdf.
function stablechar(t::Float64, mu::Float64, alfa::Float64, beta::Float64,
                    sigma::Float64)
    μ, α, β, σ = mu, alfa, beta, sigma
    t == 0.0 && return complex(1.0)
    if α != 1.0
        # Two rewrites keep the skewness term accurate through α → 1, where the
        # naive forms cancel catastrophically and the cf silently degrades
        # toward the symmetric one:
        # - expm1 form of (σ|t|)^(1-α) - 1 (σ > 0 and t ≠ 0 here, so the log
        #   argument is positive)
        # - tan(πα/2) as -cot(π(α-1)/2): α - 1 is exact near 1, while the
        #   rounding of πα/2 near π/2 costs ~5 digits in the direct tangent
        logphi = im * μ * t - σ^α * abs(t)^α *
                 (1 + im * β * sign(t) * (-1 / tan(pi * (α - 1) / 2)) *
                      expm1((1 - α) * log(σ * abs(t))))
    else
        logphi = im * μ * t - σ * abs(t) * (1 + im * β * (2 / pi) * sign(t) * log(σ * abs(t)))
    end
    return exp(logphi)
end

stablechar(t::Real, mu::Real, alfa::Real, beta::Real, sigma::Real) =
    stablechar(Float64(t), Float64(mu), Float64(alfa), Float64(beta), Float64(sigma))

function stable_pdf_fft(x::AbstractVector{<:Real}, mu::Real, alfa::Real,
                        beta::Real, sigma::Real; check::Bool = true)
    μ, α, β, σ = Float64(mu), Float64(alfa), Float64(beta), Float64(sigma)
    n = 8192

    # Pad the grid well past the data range: the FFT yields the *periodized*
    # pdf, so with heavy tails the images at ± span alias into the evaluation
    # window. Padding by the data range pushes that error below ~1e-4.
    lo, hi = extrema(x)
    pad = 2 * σ + (Float64(hi) - Float64(lo))
    lo = Float64(lo) - pad
    hi = Float64(hi) + pad
    span = (hi - lo) * n / (n - 1)   # k = 0:n-1 then covers [lo, hi] inclusive
    dx = span / n
    dt = 2 * pi / span
    T = n / 2 * dt

    j = 0:(n - 1)
    t = -T .+ j .* dt
    phi = cis.(-dt * lo .* j) .* stablechar.(t, μ, α, β, σ)
    xk = lo .+ j .* dx
    pdfk = real.(dt / (2 * pi) .* cis.(T .* xk) .* fft(phi))

    itp = Interpolations.scale(interpolate(pdfk, BSpline(Cubic(Line(OnGrid())))),
                               range(lo; step = dx, length = n))
    p = [itp(Float64(xi)) for xi in x]

    if check
        # guard against spline undershoot in regions of vanishing density
        @. p = ifelse(p <= 0, sqrt(eps(Float64)), p)
    end
    return p
end

# Adaptive-quadrature pdf based on Nolan's integral representation
# (Borak, Härdle & Weron eq. 5-8). Slower than stable_pdf but independent of
# the precomputed quadrature tables — useful for verification.
function stable_pdf_quad(x::Real, mu::Real, alpha::Real, beta::Real, sigma::Real)
    return stdstable_pdf_quad(Float64((x - mu) / sigma), Float64(alpha),
                              Float64(beta)) / Float64(sigma)
end

function stdstable_pdf_quad(x::Float64, alpha::Float64, beta::Float64)
    α, β = alpha, beta
    halfpi = pi / 2

    if α != 1.0
        ζ = -β * tan(pi * α / 2)
        ξ = atan(-ζ) / α
        c = α / (α - 1)
        if x == ζ
            return gamma(1 + 1 / α) * cos(ξ) / (pi * (1 + ζ^2)^(1 / (2 * α)))
        elseif x < ζ
            return stdstable_pdf_quad(-x, α, -β)
        else
            V(θ) = cos(α * ξ)^(1 / (α - 1)) * (cos(θ) / sin(α * (ξ + θ)))^c *
                   cos(α * ξ + (α - 1) * θ) / cos(θ)
            xc = (x - ζ)^c
            int, _ = quadgk(θ -> V(θ) * exp(-xc * V(θ)), -ξ, halfpi;
                            rtol = 1e-10, atol = 1e-13)
            return α * (x - ζ)^(1 / (α - 1)) / (pi * abs(α - 1)) * int
        end
    else
        β == 0.0 && return 1 / (pi * (1 + x^2))
        β < 0.0 && return stdstable_pdf_quad(-x, α, -β)
        V1(θ) = (2 / pi) * ((halfpi + β * θ) / cos(θ)) *
                exp((1 / β) * (halfpi + β * θ) * tan(θ))
        g1 = exp(-halfpi * x / β)
        int, _ = quadgk(θ -> V1(θ) * exp(-g1 * V1(θ)), -halfpi, halfpi;
                        rtol = 1e-10, atol = 1e-13)
        return g1 / (2 * β) * int
    end
end
