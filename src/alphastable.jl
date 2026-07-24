# The alpha-stable distribution advocated by Mandelbrot, Taleb and many others:
# https://en.wikipedia.org/wiki/Stable_distribution
#
# Numerics follow Weron, Nolan and Ament & O'Neil (qastable):
# - Weron: http://sfb649.wiwi.hu-berlin.de/papers/pdf/SFB649DP2005-008.pdf
# - Nolan: http://fs2.american.edu/jpnolan/www/stable/density.pdf
# - Ament & O'Neil: https://arxiv.org/pdf/1607.04247.pdf
#   (Julia port of https://gitlab.com/s_ament/qastable)

"""
    AlphaStable(μ, α, β, σ)

Alpha-stable distribution in Zolotarev's (M) parameterization (= Nolan's
0-parameterization, S0) with location `μ`, stability index `α ∈ (0, 2]`,
skewness `β ∈ [-1, 1]` and scale `σ > 0`.

Note that in this parameterization `α = 2` corresponds to `Normal(μ, √2 σ)`
and `α = 1, β = 0` to `Cauchy(μ, σ)`.

The pdf/cdf are computed with the quadrature and series methods of
Ament & O'Neil (qastable), which support:

- `β = 0`: `α ∈ [0.5, 2]` (pdf and cdf)
- `β ≠ 0`: `α ∈ [0.5, 0.9] ∪ [1.1, 2]` for the pdf, `α ∈ [1.1, 2]` for the cdf

Parameter combinations outside these ranges throw a `DomainError`. All
computations are performed in `Float64` (the quadrature tables are `Float64`).

```julia
d = AlphaStable(0.0, 1.5, 0.3, 2.0)
pdf(d, 0.5); cdf(d, 0.5); quantile(d, 0.99); rand(d, 1000)
fit_mle(AlphaStable, data)
```
"""
struct AlphaStable{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    α::T
    β::T
    σ::T

    function AlphaStable{T}(μ::T, α::T, β::T, σ::T;
                            check_args::Bool = true) where {T<:Real}
        if check_args
            σ > zero(σ) ||
                throw(DomainError(σ, "AlphaStable: scale σ must be positive"))
            zero(α) < α <= oftype(α, 2) ||
                throw(DomainError(α, "AlphaStable: stability index α must satisfy 0 < α ≤ 2"))
            -one(β) <= β <= one(β) ||
                throw(DomainError(β, "AlphaStable: skewness β must satisfy -1 ≤ β ≤ 1"))
        end
        return new{T}(μ, α, β, σ)
    end
end

AlphaStable(μ::T, α::T, β::T, σ::T; kwargs...) where {T<:Real} =
    AlphaStable{T}(μ, α, β, σ; kwargs...)
AlphaStable(μ::Real, α::Real, β::Real, σ::Real; kwargs...) =
    AlphaStable(promote(μ, α, β, σ)...; kwargs...)
AlphaStable(μ::Integer, α::Integer, β::Integer, σ::Integer; kwargs...) =
    AlphaStable(float(μ), float(α), float(β), float(σ); kwargs...)

AlphaStable() = AlphaStable(0.0, 2.0, 0.0, 1.0)
# standard symmetric stable variable
AlphaStable(α::Real) = AlphaStable(zero(α), α, zero(α), one(α))
# standard skewed stable variable
AlphaStable(α::Real, β::Real) = AlphaStable(zero(α), α, β, one(α))

@distr_support AlphaStable -Inf Inf

#### Conversions

Base.convert(::Type{AlphaStable{T}}, d::AlphaStable) where {T<:Real} =
    AlphaStable{T}(T(d.μ), T(d.α), T(d.β), T(d.σ); check_args = false)
Base.convert(::Type{AlphaStable{T}}, d::AlphaStable{T}) where {T<:Real} = d

#### Parameters

params(d::AlphaStable) = (d.μ, d.α, d.β, d.σ)
partype(::AlphaStable{T}) where {T<:Real} = T
location(d::AlphaStable) = d.μ
scale(d::AlphaStable) = d.σ

# ζ, the shift between Zolotarev's (M)/Nolan's S0 parameterization and the
# classical S1 parameterization: X_M = X_S1 + σζ
stable_zeta(α::Real, β::Real) = -β * tan(α * pi / 2)

#### Statistics

function mean(d::AlphaStable)
    # in the (M) parameterization E[X] = μ + σζ (finite only for α > 1)
    return d.α > 1 ? float(d.μ + d.σ * stable_zeta(d.α, d.β)) : float(oftype(d.μ, NaN))
end
var(d::AlphaStable) = d.α == 2 ? float(2 * d.σ^2) : float(oftype(d.σ, Inf))
mode(d::AlphaStable) = d.β == 0 ? float(d.μ) : float(oftype(d.μ, NaN))
median(d::AlphaStable) = d.β == 0 ? float(d.μ) : quantile(d, 1 / 2)

#### Evaluation

function pdf(d::AlphaStable, x::Real)
    μ, α, β, σ = params(d)
    s = Float64((x - μ) / σ)
    return stable_pdf(s, Float64(α), Float64(β)) / σ
end

function logpdf(d::AlphaStable, x::Real)
    p = pdf(d, x)
    return p > 0 ? log(p) : oftype(float(p), -Inf)
end

function cdf(d::AlphaStable, x::Real)
    μ, α, β, σ = params(d)
    s = Float64((x - μ) / σ)
    return clamp(stable_cdf(s, Float64(α), Float64(β)), 0.0, 1.0)
end

function quantile(d::AlphaStable, p::Real)
    0 <= p <= 1 || throw(DomainError(p, "quantile is defined for p ∈ [0, 1]"))
    p == 0 && return -Inf
    p == 1 && return Inf
    pf = Float64(p)
    lo, hi = quantile_bracket(d, pf)
    return find_zero(x -> cdf(d, x) - pf, (lo, hi))
end

# expand [μ - kσ, μ + kσ] until it brackets the target probability
function quantile_bracket(d::AlphaStable, p::Float64)
    μ = Float64(d.μ)
    σ = Float64(d.σ)
    k = 1.0
    lo = μ - k * σ
    while cdf(d, lo) > p && k < 2.0^48
        k *= 2
        lo = μ - k * σ
    end
    k = 1.0
    hi = μ + k * σ
    while cdf(d, hi) < p && k < 2.0^48
        k *= 2
        hi = μ + k * σ
    end
    return lo, hi
end

"""
    quantile(d::AlphaStable, p::AbstractVector{<:Real})

Batch quantiles. For large batches the quantile function is inverted on an
interpolated cdf grid instead of root-finding every point; probabilities are
clamped to `[1e-6, 1 - 1e-6]` on that fast path.
"""
function quantile(d::AlphaStable, p::AbstractVector{<:Real})
    length(p) < 700 && return [quantile(d, pp) for pp in p]

    p_lo = 1e-6
    pc = clamp.(Float64.(p), p_lo, 1 - p_lo)
    p_min, p_max = extrema(pc)
    σ = Float64(d.σ)
    x_min = quantile(d, p_min) - 0.1 * σ
    x_max = quantile(d, p_max) + 0.1 * σ

    steps = max(11, floor(Int, 0.8 * log2(length(p))))
    xs = range(x_min, x_max; length = 2^steps + 1)
    us = [cdf(d, x) for x in xs]
    us_inc, xs_inc = strictly_increasing_knots(us, collect(xs))

    itp = Interpolations.extrapolate(
        Interpolations.interpolate((us_inc,), xs_inc, Gridded(Linear())), Flat())
    return [itp(pp) for pp in pc]
end

# drop grid points where the numerical cdf fails to strictly increase, so the
# values can be used as interpolation knots
function strictly_increasing_knots(us::Vector{Float64}, xs::Vector{Float64})
    keep = trues(length(us))
    last_u = us[1]
    for i in 2:length(us)
        if us[i] <= last_u
            keep[i] = false
        else
            last_u = us[i]
        end
    end
    return us[keep], xs[keep]
end

cf(d::AlphaStable, t::Real) = stablechar(Float64(t), Float64.(params(d))...)

#### Sampling

# Chambers–Mallows–Stuck, cf. Weron (1996). CMS generates a standard S1
# variable; adding ζ converts it to the (M)/S0 parameterization used here.
function rand(rng::AbstractRNG, d::AlphaStable)
    μ, α, β, σ = Float64.(params(d))
    U = (rand(rng) - 0.5) * pi        # Uniform(-π/2, π/2)
    W = randexp(rng)                  # Exponential(1)
    ζ = stable_zeta(α, β)

    if α != 1.0
        θ0 = atan(-ζ) / α
        x1 = (1 + ζ^2)^(1 / (2 * α)) * sin(α * (U + θ0)) / cos(U)^(1 / α) *
             (cos(U - α * (U + θ0)) / W)^((1 - α) / α)
        return μ + σ * (x1 + ζ)
    else
        halfpi = pi / 2
        x1 = ((halfpi + β * U) * tan(U) -
              β * log(halfpi * W * cos(U) / (halfpi + β * U))) / halfpi
        # for α = 1 the standard S1 and S0 laws coincide once scaled by σ:
        # cf(σ Z₁ + μ) equals the S0 characteristic function with scale σ
        return μ + σ * x1
    end
end

#### Fast batch evaluation via FFT of the characteristic function

"""
    pdf_fft(d::AlphaStable, x::AbstractVector{<:Real})

Pdf of `d` at all points of `x` at once, via FFT inversion of the
characteristic function on a grid spanning `extrema(x)` plus padding, with
cubic-spline interpolation. Much faster than `pdf.(d, x)` for large `x`;
accuracy is limited by the grid resolution.
"""
pdf_fft(d::AlphaStable, x::AbstractVector{<:Real}) =
    stable_pdf_fft(x, params(d)...)

"""
    logpdf_fft(d::AlphaStable, x::AbstractVector{<:Real})

`log.(pdf_fft(d, x))`, with nonpositive interpolated densities floored to
`√eps` so the result stays finite — intended for likelihood maximization.
"""
logpdf_fft(d::AlphaStable, x::AbstractVector{<:Real}) = log.(pdf_fft(d, x))
