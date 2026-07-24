# Stable.jl

[![CI](https://github.com/haavardhvarnes/Stable.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/haavardhvarnes/Stable.jl/actions/workflows/CI.yml)

An extension of [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
for the [α-stable distribution](https://en.wikipedia.org/wiki/Stable_distribution) —
the heavy-tailed family advocated by Mandelbrot, Taleb and many others for
modeling financial returns and other extreme-event data.

```julia
using Stable

d = AlphaStable(0.0, 1.5, 0.3, 2.0)   # μ, α, β, σ

pdf(d, 0.5)
cdf(d, 0.5)
quantile(d, 0.99)
cf(d, 0.7)                # characteristic function
x = rand(d, 10_000)       # Chambers–Mallows–Stuck sampling

fit_mle(AlphaStable, x)   # maximum likelihood (McCulloch start + Ipopt)
```

## Installation

Registered in the personal registry at
[`haavardhvarnes/JuliaRegistry`](https://github.com/haavardhvarnes/JuliaRegistry).
From a fresh Julia REPL:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/haavardhvarnes/JuliaRegistry"))
Pkg.add("Stable")
```

## Parameterization

`AlphaStable(μ, α, β, σ)` uses **Zolotarev's (M) parameterization (= Nolan's S0)
consistently** across pdf, cdf, cf, sampling, moments and fitting. In this
parameterization:

- `α = 2` is `Normal(μ, √2 σ)`,
- `α = 1, β = 0` is `Cauchy(μ, σ)`,
- `mean(d) = μ + σζ` for `α > 1`, with `ζ = -β tan(πα/2)`.

Software using the classical S1 parameterization (e.g. scipy, R stabledist)
differs by the location shift `σζ`: `X_M = X_S1 + σζ`.

## Supported parameter ranges

The pdf/cdf use the quadrature + series methods of
[Ament & O'Neil (qastable)](https://gitlab.com/s_ament/qastable), which cover:

| case | pdf | cdf / quantile |
|---|---|---|
| `β = 0` | `α ∈ [0.5, 2]` | `α ∈ [0.5, 2]` |
| `β ≠ 0` | `α ∈ [0.5, 0.9] ∪ [1.1, 2]` | `α ∈ [1.1, 2]` |

Outside these ranges a `DomainError` is thrown. All numerics run in `Float64`.

## Fitting

- `fit_mle(AlphaStable, x)` / `fitstable(x)` — maximum likelihood: McCulloch's
  (1986) quantile estimator seeds an Ipopt run maximizing an FFT-based
  log-likelihood; falls back to the McCulloch estimate if the solver fails.
  Restricted to `α ∈ [1.11, 2)`.
- `fitcullstable(x)` — the McCulloch quantile estimator alone (fast, consistent).
- `fitmlestable(x)` — the underlying routine, returning
  `(params, loglike, status)`.
- `refitstable(d, x)` — warm-started refit.

## Batch evaluation

`pdf.(d, xs)` broadcasts the pointwise quadrature. For large batches,
`pdf_fft(d, xs)` / `logpdf_fft(d, xs)` invert the characteristic function with
a single FFT (accuracy ≈ 1e-4); `quantile(d, ps)` switches to an interpolated
inverse-cdf grid for 700+ probabilities.

## Testing

```julia
using Pkg; Pkg.test("Stable")
```

The suite (700+ assertions) checks the quadrature tables against an
independent adaptive-quadrature oracle built on Nolan's integral
representation, Kolmogorov–Smirnov tests of the sampler against the cdf, cf ↔
pdf Fourier consistency, moment formulas against numerical integrals, and
parameter recovery of the estimators.

## References

- Ament & O'Neil (2017), *Accurate and efficient numerical calculation of
  stable densities via optimized quadrature and asymptotics*,
  [arXiv:1607.04247](https://arxiv.org/pdf/1607.04247.pdf) —
  [qastable](https://gitlab.com/s_ament/qastable), origin of the core numerics.
- Nolan, *Numerical calculation of stable densities and distribution
  functions* — [pdf](http://fs2.american.edu/jpnolan/www/stable/density.pdf).
- Weron (2004), *Computationally intensive Value at Risk calculations* —
  [pdf](http://sfb649.wiwi.hu-berlin.de/papers/pdf/SFB649DP2005-008.pdf).
- Borak, Härdle & Weron (2005), *Stable distributions* —
  [pdf](http://prac.im.pwr.edu.pl/~hugo/publ/SFB2005-008_Borak_Haerdle_Weron.pdf).
- McCulloch (1986), *Simple consistent estimators of stable distribution
  parameters*.
