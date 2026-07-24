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

# Grid design: work on the standardized law, restrict the FFT/spline to the
# body window between the two tail-series thresholds (points beyond them get
# the certified series instead of spline extrapolation), size the frequency
# range from the known cf decay |cf(t)| = exp(-t^α), and pad the spatial span
# so that the FFT's periodized tail images alias below ~1e-6. This keeps the
# grid size data-independent (the body window is threshold-capped) and lifts
# the accuracy of the old data-spanning grid (~1e-4) to ~1e-6 with exact
# relative accuracy in the tails.
function stable_pdf_fft(x::AbstractVector{<:Real}, mu::Real, alfa::Real,
                        beta::Real, sigma::Real; check::Bool = true)
    μ, α, β, σ = Float64(mu), Float64(alfa), Float64(beta), Float64(sigma)
    (0.0 < α <= 2.0 && σ > 0.0) ||
        throw(DomainError((α, σ), "stable_pdf_fft requires 0 < α ≤ 2 and σ > 0"))
    isempty(x) && return Float64[]

    n_inf = pdf_series_terms(α, β)
    ζ = stable_zeta(α, β)
    # at exactly α = 1 with β ≠ 0, ζ = -β·tan(π/2) ≈ ∓8e15 is a floating-point
    # artifact and the (M)-form tail series in (x - ζ) is meaningless; use a
    # fixed window and a dedicated tail evaluation instead
    alpha_one_skewed = α == 1.0 && β != 0.0
    if alpha_one_skewed
        upper = 50.0
        lower = -50.0
    else
        upper = tail_series_threshold(α, ζ, n_inf)
        lower = -tail_series_threshold(α, -ζ, n_inf)
    end

    out = Vector{Float64}(undef, length(x))
    body_idx = Int[]
    body_s = Float64[]
    for (i, xi) in pairs(x)
        s = (Float64(xi) - μ) / σ
        if s > upper
            out[i] = (alpha_one_skewed ? alpha_one_skewed_tail_pdf(s, β) :
                      stable_pdf_series_infinity(s, α, β, n_inf)) / σ
        elseif s < lower
            out[i] = (alpha_one_skewed ? alpha_one_skewed_tail_pdf(s, β) :
                      stable_pdf_series_infinity(-s, α, -β, n_inf)) / σ
        else
            push!(body_idx, i)
            push!(body_s, s)
        end
    end

    if !isempty(body_idx)
        w_lo = minimum(body_s) - 0.5
        w_hi = maximum(body_s) + 0.5

        # frequency range needed for the cf to decay below ~1e-16, and a grid
        # step fine enough for ~1e-6 cubic-spline accuracy
        t_need = (-log(1e-16))^(1 / α)
        dx_target = min(0.1, pi / t_need)
        # spatial padding so the periodized tail images stay below ~1e-6
        c_tail = α * max(tail_cdf_coefficient(α, β), tail_cdf_coefficient(α, -β))
        pad = clamp((max(c_tail, 1e-4) / 1e-6)^(1 / (1 + α)), 8.0, 4000.0)

        # the 2^20 cap is only reached for α ≲ 0.7; below α ≈ 0.5 even that
        # grid cannot resolve the cf decay and accuracy degrades (documented)
        span = (w_hi - w_lo) + 2 * pad
        n = clamp(nextpow(2, ceil(Int, span / dx_target)), 1024, 1 << 20)
        dx = span / n
        dt = 2 * pi / span
        T = n / 2 * dt
        g_lo = w_lo - pad

        j = 0:(n - 1)
        t = -T .+ j .* dt
        phi = cis.(-dt * g_lo .* j) .* stablechar.(t, 0.0, α, β, 1.0)
        sk = g_lo .+ j .* dx
        pdfk = real.(dt / (2 * pi) .* cis.(T .* sk) .* fft(phi))

        itp = Interpolations.scale(interpolate(pdfk, BSpline(Cubic(Line(OnGrid())))),
                                   range(g_lo; step = dx, length = n))
        for (b, s) in enumerate(body_s)
            out[body_idx[b]] = itp(s) / σ
        end
    end

    if check
        # guard against spline undershoot in regions of vanishing density
        @. out = ifelse(out <= 0, sqrt(eps(Float64)), out)
    end
    return out
end

# Pdf for α = 1 with β ≠ 0 outside the FFT window: adaptive inversion of the
# α = 1 characteristic function for moderate |s|, and the leading tail
# asymptote (1 ± β)/(π s²) beyond |s| = 2000 (relative accuracy ~1e-3 there).
function alpha_one_skewed_tail_pdf(s::Float64, β::Float64)
    sa = abs(s)
    βs = s >= 0 ? β : -β    # reflection: f(-s; 1, β) = f(s; 1, -β)
    sa > 2000.0 && return (1 + βs) / (pi * sa^2)
    f(t) = t == 0.0 ? 1.0 : cos(sa * t + βs * (2 / pi) * t * log(t)) * exp(-t)
    int, _ = quadgk(f, 0.0, 40.0; rtol = 1e-10, atol = 1e-300)
    return int / pi
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
