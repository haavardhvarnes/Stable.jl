# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Julia package (`Stable`) extending Distributions.jl with the α-stable distribution:
`AlphaStable{T} <: ContinuousUnivariateDistribution` with parameters `(μ, α, β, σ)`.
**Zolotarev's (M) parameterization (= Nolan's S0) is used consistently throughout** — pdf, cdf,
cf, `rand`, `mean` and the fitted parameters all agree on it. Keep that invariant when touching
anything: in this parameterization `α = 2` is `Normal(μ, √2σ)` and `α = 1, β = 0` is
`Cauchy(μ, σ)`, and other software (scipy, R stabledist, Nolan's S1) may differ by the shift
`σζ` where `ζ = -β tan(πα/2)`.

The numerics are a Julia port of Ament & O'Neil's *qastable* MATLAB code
(https://gitlab.com/s_ament/qastable), following:

- Weron: http://sfb649.wiwi.hu-berlin.de/papers/pdf/SFB649DP2005-008.pdf
- Nolan: http://fs2.american.edu/jpnolan/www/stable/density.pdf
- Ament & O'Neil: https://arxiv.org/pdf/1607.04247.pdf
- Borak, Härdle & Weron: http://prac.im.pwr.edu.pl/~hugo/publ/SFB2005-008_Borak_Haerdle_Weron.pdf

## Commands

- Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'` (or directly:
  `julia --project=. test/runtests.jl`; Test is a stdlib so both work)
- Instantiate deps after checkout: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Quick REPL session: `julia --project=.` then `using Stable`
- There is no separate build/lint step. Format with Runic.jl if installed (it currently is not).
- GitHub: `haavardhvarnes/Stable.jl` (private); CI (`.github/workflows/CI.yml`) runs the test
  suite on Julia LTS + stable. The committed Manifest pins the maintainer's resolve; CI deletes
  it before instantiating.
- Registered in the personal registry `haavardhvarnes/JuliaRegistry` (clone at
  `~/.julia/registries/JuliaRegistry`). To release a new version: bump `version` in
  Project.toml, commit and push, then
  `julia -e 'using LocalRegistry; register("."; registry = "JuliaRegistry", push = true)'`.

## Supported parameter ranges (throw `DomainError` outside)

- `β = 0`: `α ∈ [0.5, 2]` for pdf and cdf
- `β ≠ 0`: pdf `α ∈ [0.5, 0.9] ∪ [1.1, 2]`, cdf (and hence quantile) `α ∈ [1.1, 2]`
- MLE fitting restricts to `α ∈ [1.11, 2)` for this reason.

All numerics run in `Float64` (the quadrature node tables are Float64); the struct itself is
generic in `T` and inputs are converted.

## Architecture

Three layers; the standard-law numerics (μ = 0, σ = 1) are separate from the
Distributions.jl interface:

1. **Standard-law numerics** — `stable_pdf.jl` / `stable_cdf.jl` are dispatchers picking per
   region: symmetric closed forms (α = 2, α = 1) → reflection identity (`x < ζ` maps to
   `(-x, -β)`) → tail series above a computed `min_inf_x` threshold
   (`*_series_infinity.jl`, computed via `loggamma` ratios to stay in Float64) → otherwise a
   fixed-node Fourier-integral quadrature (`*_integral.jl` files, hard-coded qastable
   nodes/weights stored as `const GX_*`/`GW_*` vectors). `stable_pdf_series_zero.jl` and the
   three `*_integrand.jl` files are reference implementations, not wired into the dispatchers.
2. **Distributions.jl interface** — `alphastable.jl`: struct + validation, `pdf` (standardize,
   evaluate, divide by σ), `cdf`, `logpdf`, scalar `quantile` (bracket expansion + `find_zero`),
   batch `quantile` on ≥ 700 points via an interpolated inverse-cdf grid, `cf`, and `rand`
   (Chambers–Mallows–Stuck **plus the `+ζ` shift converting the S1 draw to S0**).
   `stable_pdf_fft.jl` holds the shared characteristic function `stablechar` (single source of
   truth for `cf`), the FFT-inversion batch pdf (`pdf_fft`/`logpdf_fft`, grid padded by the data
   range because the FFT periodizes the heavy-tailed pdf — accuracy ~1e-4), and
   `stdstable_pdf_quad`, an independent QuadGK/Nolan-integral pdf used as the test oracle.
3. **Fitting** — `fit.jl`: `fitcullstable` (McCulloch quantile estimator; his ζ **is** the S0
   location, so it is returned as μ directly) seeds `fitmlestable` (JuMP 1.x + Ipopt maximizing
   the FFT-based log-likelihood with central finite-difference gradients,
   `hessian_approximation = limited-memory`, tolerances deliberately loose to match FD noise —
   tightening them just makes Ipopt run to the iteration cap). Falls back to the McCulloch
   estimate if the solver fails or worsens the likelihood. `fit_mle(AlphaStable, x)` /
   `fitstable` / `refitstable` wrap it.

## Deliberate design decisions (do not "fix" these back)

- S0 consistency fixes applied during the 2026 modernization, all verified in tests against
  quadrature/sampling: pdf divides by σ; `rand` adds ζ and uses the `1/(2α)` CMS exponent;
  `mean = μ + σζ` for α > 1; the α = 2 branches use variance 2σ²; the symmetric cdf takes the
  complement in the deep left tail; McCulloch μ is not converted to S1.
- Unsupported (α, β) regions throw `DomainError` (the older sibling code printed and returned
  NaN); the fitting objective avoids them by clamping and using `pdf_fft`, which has no range
  restriction.
- Vectorized `pdf(d, xs)`/`logpdf(d, xs)` methods were dropped — use broadcasting
  (`pdf.(d, xs)`) or `pdf_fft`/`logpdf_fft` for large batches. The batch `quantile` method was
  kept because it is a genuinely different (grid) algorithm.
- Dierckx was replaced by Interpolations (already a dep) for the FFT grid spline; the local SQP
  solver used by the sibling version was not ported — Ipopt is registered and sufficient.
- `test/MonthRolling.csv` is legacy sample data for manual experiments; the test suite does not
  use it.

## Prior versions elsewhere (context — do not edit those folders)

Earlier attempts live under `../`; their fixes were ported here in July 2026, so they are now
reference-only:

- `../Distributions/AlphaStable/` — the source of the ported fixes (flat include style, FFT pdf,
  SQP fitting). Superseded by this repo.
- `../Distributions/Stable/` — stale byte-copy of this repo's pre-modernization state.
- `../Distributions/qastable/` — original MATLAB sources (`mfiles/`) plus a direct Julia port;
  the authority when a quadrature/series port is in doubt.
- `../StableDist/StandardStable/` — separate Borak/Härdle/Weron FFT attempt (`borak.jl`,
  `weronfft.jl`).
- Reference papers (PDFs) in `../Distributions/docs/` and `../StableDist/docs/`.
