# Stable.jl acceleration report: computational harmonic analysis 2015–2026, verified against this codebase

All numbers below are either (a) measured on the maintainer's machine (Apple M1, Julia 1.12.6, this repo) during verification, (b) taken from primary sources read in full, or (c) estimates — every (c) and every claim carrying only a PARTLY_CONFIRMED or UNVERIFIABLE verdict is flagged inline.

**Measured baselines (this machine):**

| Path | Cost | Accuracy |
|---|---|---|
| Scalar pdf/cdf (39–94-node rule) | 1.4–3.4 µs/pt (measured 3.6 µs) | max abs err 7.5e-16–8.3e-15 in supported region |
| Batch `pdf_fft` (8192-pt FFT + cubic spline) | 1.44 ms / 10k pts | ~1e-4 (aliasing + spline floor) |
| `negloglike` (n=4000) | 1.62 ms/call = 0.90 cf-grid + 0.88 spline + 0.17 FFT | inherits ~1e-4; O(1) *relative* logpdf error at tail samples |
| MLE fit | 0.72 s warm (~441 objective calls, central-FD gradients = 8 nll calls each); 4–5 s cold is mostly JuMP/Ipopt compile | — |
| Scalar quantile | ~200 µs (~70 cdf calls, bracket + bisection) | — |
| Batch quantile | 7.5 ms fixed (2049 cdf calls) + 0.7 µs/pt | linear-interp; p clamped to [1e-6, 1-1e-6] |
| CMS `rand` | 59.6 ns/draw scalar | — |

---

## 1. Ranked shortlist

### 1. Hoist + SIMD-batch the existing fixed-node quadrature kernel; route batch pdf and the MLE objective through it — **CONFIRMED**

**What.** `src/stable_pdf_fourier_integral.jl` recomputes `rank_scaling` (log+pow), `z = -β·tan(απ/2)`, and per-node `gxⱼ^α`, `exp(-gxⱼ^α)` on *every call*, though none depend on x. Hoist them into cached per-(α,β) vectors; the per-point work collapses to `acc += wⱼ·cos((x-ζ)·tⱼ + uⱼ)` — one cos and two FMAs per node. Then batch over x with `@turbo` (LoopVectorization + SLEEFPirates). Identical structure applies to the cdf's sin/t kernel.

**Gain (independently re-measured, 86 nodes).** 2815 ns/pt naive → 400 ns hoist-only (7.0x, zero new dependencies) → 143 ns `@turbo`-batched (19.7x) → 42.7 ns/pt on 4 threads; maxdiff vs Base 4.2e-17; SLEEFPirates cos verified 1 ulp for |arg| ≤ 1e6 (covers scaled nodes to t~1100 at α=0.5). Consequences: 10k-point batch 1.44 ms → 427 µs (3.4x) **at quadrature accuracy instead of 1e-4** (verified 1e-8 vs oracle; ~1e-13 by design), eliminating aliasing floor, spline stage, and the `sqrt(eps)` logpdf flat spots; MLE objective 1.62 → 0.2–0.8 ms, warm fit 0.72 s → ~0.15–0.4 s with no gradient-code changes.

**Effort.** Days. **Risk.** Low. LoopVectorization/SLEEFPirates are grant-maintained post-Elrod; hedges are real (hoist-only 7x is pure Base; SIMDMathFunctions.jl is a registered fallback). Cached vectors need a home in the immutable struct (constructor-time ~2–3 µs build, or lazy Ref). Does not close the α∈(0.9,1.1), β≠0 gap. Batch API must partition points across the tail-series threshold as the scalar path already does.

**First step.** Restructure the two integral files into `setup(α,β) → cached vectors` + `@turbo` point×node loop; validate against `stable_pdf_quad` at rtol 1e-8. Working benchmark skeleton exists in the session scratchpad (`simdbench.jl`).

**Citations.** https://arxiv.org/abs/2001.09258 (SLEEF, IEEE TPDS 2020); https://github.com/JuliaSIMD/LoopVectorization.jl; https://github.com/ClimFlows/SIMDMathFunctions.jl; https://julialinearalgebra.github.io/AppleAccelerate.jl/dev/benchmarks/ (optional macOS backend).

### 2. Newton quantiles on the existing kernel + analytic tail inversion + Junike error control — **confirmed by local benchmark; theory published**

**What.** Replace bracket + `find_zero` (~70 cdf calls) with bisection-safeguarded Newton using the pdf as the exact derivative, seeded by the bracket, terminated via Junike's method-agnostic Theorem 2.1 (|F⁻¹(p) − H⁻¹(p)| ≤ 2ε/f(q) + ε — applies directly to the existing quadrature cdf), and hand extreme p to closed-form inversion of the already-shipped Zolotarev tail series, Q(p) ≈ ζ + c·p^(−1/α).

**Gain.** 200 µs → 15–40 µs/quantile (5–13x) at full accuracy today; ~2–5 µs (40–100x) after item 1 speeds each cdf/pdf call. Removes the two ~200 µs edge-quantile calls from the batch setup; extreme-p quantiles become exact instead of clamped at 1e-6. Interim one-liner: `Roots.ITP()` on the existing bracket gives ~3–5x.

**Effort.** Hours (~50–100 lines, zero new deps). **Risk.** Lowest of the list: needs a bisection fallback where the pdf underflows and an oracle-validated handoff to the tail inverse; per-quantile cost grows as 1/f(q) toward the tails (bounded by the tail takeover).

**First step.** Swap the solver at `src/alphastable.jl:122`; test over α∈{0.6,1.2,1.5,1.9} × β∈{0,0.5,0.9} × p∈[1e-10, 1−1e-10].

**Citations.** https://arxiv.org/abs/2502.13537 (Junike 2025, Statistics & Probability Letters; Thm 2.1, adaptive-ε loop).

### 3. FFT-path repairs: cf-decay-driven grid sizing + Bergström tail splicing — **CONFIRMED (the surviving parts of a four-part idea)**

**What.** (a) Size T from the exactly known cf decay, T_need = (ln 1/ε)^(1/α)/σ, instead of the fixed n=8192/data-span coupling — the current grid over-provisions T by 5–30x (actual T=207 vs needed 7.0 at α=1.5, data range 40). (b) Route |x−ζ| > `min_inf_x` points through the already-shipped Bergström tail series instead of the spline + `p≤0 → sqrt(eps)` guard.

**Gain.** Batch 1.44 → ~0.5–0.8 ms (cf-grid and spline stages shrink with N: 8192 → ~1024–2048) and body accuracy ~1e-4 → ~1e-6 with certified series tails. This fixes the single worst MLE pathology: 1e-4 *absolute* error is O(1) *relative* error where tail samples live (pdf ~3e-5 at |x|~300, α=1.5, n=5000). nll 1.62 → ~0.6–0.9 ms, warm fit ~2x, zero new dependencies. Keep this even after item 1 lands — it is the fallback path for α near 1, β≠0.

**Effort.** Hours (~15 + 30–60 lines). **Risk.** Near-zero; reuse the existing oracle-tested `min_inf_x` thresholds (the series is asymptotic for α>1 — do not invent new splice points). Keep the current grid for α ≲ 1 where T_need explodes.

**Why not Simpson weights or PROJ filters.** Refuted for this code: the endpoint cf is ~1e-300–1e-14, so the left-rectangle sum already *is* the trapezoid/Poisson-exact sum; the error is aliasing + spline, not quadrature order (and d⁴/dt⁴ e^{−|t|^α} ~ |t|^{α−4} voids Simpson's O(h⁴) at 0 anyway). The stable density is C^∞ — there is no Gibbs for spectral filters to fix, and filters do not remove aliasing.

**Citations.** https://arxiv.org/abs/2303.16012 (Junike, Numer. Math. 156:533–564, 2024 — Thm 3.8 closed-form L,N; Thm 4.1(ii) sharp O(N^−α) rate); Menn & Rachev, Comput. Stat. Data Anal. 50(8):1891–1904, 2006 (FFT-center + Bergström-tail splicing precedent, https://dl.acm.org/doi/10.1016/j.csda.2005.03.004); SWIFT bandwidth rule: Ortiz-Gracia & Oosterlee, SIAM J. Sci. Comput. 38(1), 2016.

### 4. Characteristic-function-domain TMLE as the default fast fit — **[PARTLY_CONFIRMED: algorithm verified from the paper; wall-clock is a projection; unpublished preprint]**

**What.** Matsui–Sueishi projected-score estimator: score projected onto {cos(uᵢx), sin(uᵢx)}, k=101 fixed u-points; empirical cf computed once (O(nk), ~5–8 ms at n=4000); Σ(θ) has closed-form entries from the cf at uᵢ±uⱼ and is provably positive definite; damped scoring from the existing McCulloch start. No density inversion anywhere in the loop — the t=0 cf kink and slow cf decay become irrelevant; extreme observations enter only through bounded cos/sin.

**Gain.** Projected ~1.5–2 ms/iteration × 10–30 iterations → **30–60 ms total fit, 15–25x vs the 0.72 s warm baseline** (>100x vs the 4–5 s cold figure, which is mostly compile — TMLE also removes JuMP/Ipopt from the hot path). Statistical efficiency (paper Table 2, n=1000): sd ratio vs exact ML 0.986–1.006 for α∈[1.0,1.6]; degrades to 0.82–0.91 at α=1.9. **[projection, not measured end-to-end]**

**Effort.** Days (~150–250 lines, LinearAlgebra only). **Risk.** Highest on this list: still an unpublished arXiv preprint as of 2026-07, simulations only at n=1000; Fisher information diverges at α→2, |β|→1 (condition the scoring matrix, mirroring the existing β-damping); fixed u-grid assumes unit scale (standardize by McCulloch σ̂); multiple roots possible (damped steps + never-worse-than-start guard). Lower-risk fallback delivering ~2–2.3x: analytic cf-parameter-derivative gradients through the shared cf-grid/FFT/spline pipeline (gradient 5.6 ms vs FD 13 ms, measured components), which also lets Ipopt's tol tighten below 1e-4. Note: the "FD noise" motivation is refuted — the FFT bias is *smooth* in θ; analytic gradients buy speed and consistency, not accuracy.

**First step.** `src/fit_tmle.jl`; validate bias/sd/wall-time vs `fitmlestable` at n∈{2500,5000}, α∈{1.2,1.5,1.8,1.95}, β∈{0,0.5,0.9}.

**Citations.** https://arxiv.org/abs/2209.08980 (read in full: eqs 2.3–2.8, Lemma 2.2, Tables 1–2).

### 5. Type-3 NUFFT (FINUFFT.jl) with graded-panel Gauss–Legendre quadrature — **[PARTLY_CONFIRMED: working local prototype; two headline sub-claims refuted]**

**What.** Evaluate f(xₖ) = (1/π)Σⱼ wⱼ Re[e^{−i tⱼ xₖ} cf(tⱼ)] as a *true quadrature* (panels geometrically graded into the t=0 kink, uniform oscillation-resolving panels beyond, truncated where exp(−(σt)^α)<ε) with one type-3 NUFFT to the raw clamped sample points. No periodization → no aliasing floor; no spline.

**Gain (measured prototype, 10k scattered targets, tol 1e-10, vs QuadGK 1e-13 oracle).** α=1.5: 432 nodes, max err 1.63e-11, 1.09 ms one-shot, **0.150 ms plan re-exec**; α=0.7: 3616 nodes, 6.2e-12, 1.25/0.228 ms; α=0.5: 26.5k nodes, 1.75e-11, 2.5/0.87 ms. Net: parity-to-2x-slower than the current FFT path one-shot, 1.7–10x faster with plan reuse, at **7 orders of magnitude better accuracy**. `ntrans`-vectorized execution lets analytic d/dα, d/dβ strength vectors share the spreading step — near-free likelihood gradients.

**Refuted sub-claims.** "~1e-14 achievable" — the type-3 floor is (space-bandwidth product)×1e-16 ≈ 3e-14–5e-12 here; realistic certified accuracy 1e-11–1e-12. "Docs promise 10–100x plan reuse" — no such figure exists; measured 2.7–7x. Also: MLE plan reuse is weaker than hoped (α moves the nodes, µ/σ move standardized targets → `setpts!` re-runs; realistic per-eval is the full 1.1–2.5 ms).

**Effort.** Days. **Risk.** Moderate. Clamping targets at the existing tail-series switch is load-bearing (unclamped α=0.7 samples give nf~1e7–1e8 internal grids and an ~1e-8 floor); α ≤ 0.4 needs 1e5+ nodes (keep scalar/series paths there); set `nthreads=1` below ~1e5 points on M1; jll dependency (Apache-2.0, Apple Silicon binaries present). **Ranked fifth because item 1 captures most of the speed more cheaply; this is the accuracy play** — adopt if a certified ~1e-11 batch/likelihood mode is worth a binary dependency.

**Citations.** https://finufft.readthedocs.io/en/latest/tutorial/contft.html; https://arxiv.org/abs/1808.06736 (FINUFFT, SISC 2019); https://finufft.readthedocs.io/en/latest/trouble.html (error floor); https://github.com/ludvigak/FINUFFT.jl; Andersen & Lake, SSRN 4335916 (2022/23) **[UNVERIFIABLE: full text paywalled; abstract + citing papers only]**; https://arxiv.org/abs/2606.22970 (Potts–Tasche 2026, a-priori l∞ certification bounds).

---

## 2. Per-code-path analysis

### 2.1 Scalar pdf/cdf (`stable_pdf_fourier_integral.jl`, `stable_cdf_integral.jl`; 1.4–3.4 µs/pt)

**Recommended.**
- **Hoist + SIMD kernel** (shortlist #1): 1.4–3.4 µs → 0.4 µs dep-free, 0.143 µs batched, 43 ns threaded. CONFIRMED.
- **Wire the dormant Saenko near-zero series** (`stable_pdf_series_zero.jl` exists, unwired, already has the form-C→S0 mapping) with the closed-form certified threshold x_ε^N = (aπ·ε·N!/Γ((N+1)/a))^{1/N}, plus a near-zero cdf branch (none exists). Estimated 30–150 ns/pt for N=10–30 in-radius — **[estimate, not benchmarked]**; after #1 lands the speed delta shrinks to ~3–10x, so this earns its place on *certified accuracy, coverage, and a quadrature-independent second oracle*, not speed. **[PARTLY_CONFIRMED]**: bounds control truncation only — Float64 cancellation caps usable radii well below the paper's N=100–300 curves (max-term/result must stay ≲1e3); for α<1 the near-zero series is asymptotic with radius only ~0.1 at α=0.7. Citations: https://arxiv.org/abs/2210.06920, https://arxiv.org/abs/2303.03016, https://arxiv.org/abs/2303.12488.

**Honorable mentions.**
- **Boyarchenko–Levendorskii conic-contour trapezoid**: reproduced locally to 1e-13–1e-16 abs error at N=141–354 nodes, 3.1–3.8 µs/pt with precomputed tables — *parity* with the existing kernel where it works; the actual payoff is coverage (α=0.3, β=0.9 to 1.9e-15; see §2.4). BL's "several dozen nodes" is 2–4x optimistic; "parameter-uniform through α=1" overstated (admissible rotation cone width 0.032 rad at α=1.01, fails at 1.001 without curved contours). **[PARTLY_CONFIRMED]** https://arxiv.org/abs/1808.04321
- **Adaptive Levin-TSVD as an offline oracle**: reimplemented in ~60 lines; 4e-14–3e-13 abs error across the α~1 gap and branch point, and 285x faster than QuadGK at rtol 1e-13 for oscillatory (large-x) cases — but 2.1–4.6 ms/pt at runtime (the claimed 10–100 µs is 10–40x optimistic for a from-scratch semi-infinite implementation). Use for table regeneration/validation, never online. **[PARTLY_CONFIRMED]** https://arxiv.org/abs/2211.13400
- **QuadGK weighted-Gauss rules** w.r.t. e^{−t^α}: valid for the symmetric case only (β=0); ~5–15 nodes near the mode vs 43 global, per-α construction ~ms. Marginal after #1.

**Rejected.**
- **PathFinder / NSD automation**: scope is explicitly "f entire, g polynomial" — the t^α phase and exp(−t^α) amplitude both have branch points; the authors call this open research. REFUTED for direct use. https://arxiv.org/html/2307.07261
- **GeneralizedGauss.jl / Huybrechs guaranteed construction for regenerating the tables**: the guarantee holds only for *complete Chebyshev sets*; the author's own conclusion excludes highly oscillatory families like this integrand; the package is unregistered, v0.0.2, and its README calls the method "not at all efficient". A BigFloat BGR-style re-run is a weeks-scale project with a ~1.5–2.5x node-count ceiling. https://arxiv.org/abs/1710.11244
- **Complex-node oscillatory-Gauss rules**: existence only empirical (even n), ill-conditioned per-ω construction, and requires analytic continuation of t^α terms near the branch point where it is invalid.
- **Ooura–Mori DE-Fourier**: inapplicable for β≠0 (nonlinear phase breaks the zero-alignment its tail cancellation needs); symmetric case is already well served.

### 2.2 Batch pdf + MLE likelihood (`stable_pdf_fft.jl`, `fit.jl`; 1.44 ms/10k @ ~1e-4; 0.72 s warm fit)

**Recommended (in order).**
1. **Reroute through the SIMD quadrature kernel** (shortlist #1): 427 µs/10k pts on 4 threads at quadrature accuracy; MLE objective 0.2–0.8 ms exact and smooth. The fit domain α∈[1.11,2) is fully covered by the 86-node table, so the quadrature's (0.9,1.1) hole is irrelevant for MLE.
2. **FFT-path repairs** (shortlist #3) as the retained fallback: hours of work, ~2x nll, tails fixed.
3. **TMLE** (shortlist #4) if fit latency matters after 1–2: 30–60 ms projected. **[PARTLY_CONFIRMED]**
4. **Type-3 NUFFT** (shortlist #5) if a certified 1e-11 batch mode is wanted. **[PARTLY_CONFIRMED]**
5. **Analytic cf-derivative gradients through the existing pipeline** as the low-risk TMLE alternative: gradient 5.6 ms vs 13 ms FD → fit 0.72 → ~0.3–0.35 s; the spline is linear in grid values so the surrogate gradient is exact; force derivative 0 at the p≤0 guard's flat spots.

**Honorable mentions.**
- **NUFFT-COS (type-2, Le Floc'h 2025, pure-Julia NFFT.jl)**: real precedent — his measured 19–73x over classic COS, breakeven ~100 targets; for stable laws my sizing gives 5–10x over the current path at equal ~1e-4 accuracy for α≥1. But the O(N^−α) heavy-tail wall (Junike 2024, sharp) means it moves *along* the same accuracy wall the FFT path sits on, not past it, and small α blows up N (~1e5 at α=0.6). Dominated by #1 (more speed, more accuracy, no new dep) and #5 (more accuracy). His range-selection claim via Junike–Pankrashkin Cor. 9 is wrong for stable laws (needs finite even moments) — use Nolan tail asymptotics for L. **[PARTLY_CONFIRMED]** https://arxiv.org/abs/2507.13186; https://arxiv.org/abs/2208.00049
- **Low-rank tensor Chebyshev surrogate in (x,α,β)** for fitting: theory solid (Gass–Glau et al., https://arxiv.org/abs/1505.04648); honest accounting gives ~5x per objective plus analytic parameter gradients after a one-time ~0.2–2 s build at ~1e-6–1e-8 accuracy. Deferred: TMLE + #1 reach further for less machinery; α-direction analyticity through 1 requires an expm1-stable kernel and empirical degree checks. **[PARTLY_CONFIRMED, projections mine]**
- **Threaded pointwise batch (libstable architecture pattern)**: subsumed by #1's threading. libstable is GPL-2/3 — design inspiration only.
- **Potts–Tasche 2026** l∞ bounds: certification layer for whichever NUFFT/NFFT batch path ships. https://arxiv.org/abs/2606.22970

**Rejected.**
- **Simpson/Newton–Cotes FFT weights (Wang–Zhang 2008 / SciPy "fft-simpson")**: refuted for this code — see shortlist #3. Also SciPy's *default* is Nolan quadrature, not fft-simpson, and its docs call the FFT path experimental and poor for α≤1.
- **PROJ dual-frame projection + spectral filters**: no Gibbs to fix (C^∞ density); negligible gain over the existing tridiagonal spline prefilter; aliasing untouched.
- **SWIFT (full method)**: more cf evaluations than COS when both are tuned (documented 2^16–2^21 blowups in corner cases); only its 3-line cf-decay bandwidth rule survives — absorbed into shortlist #3(a).
- **COS density recovery as the batch engine**: O(N^−α) sharp ⇒ N~1e4 for 1e-6 at α=1.5, unusable ≤α≈1.2, admissibility unproven for α≤1/2; series goes negative in tails → NaN logpdf.
- **Damped COS / Carr–Madan / Lord–Kahl damping**: require exponential moments; stable laws with |β|<1 have none.
- **Fractional FFT / czt grid decoupling**: legitimate (2–30x fewer cf samples for α≥1) but most of the gain is captured with zero new code by shrinking the plain FFT after shortlist #3(a); Chourdakis' "45x" and "128-pt ≈ 4096-pt" are **[UNVERIFIABLE — full text unreachable]** and option-pricing-specific. Revisit only if the target becomes ~1e-7 with fine dx and small T simultaneously.

### 2.3 Quantile (scalar ~200 µs; batch 7.5 ms + 0.7 µs/pt)

**Recommended (compounding).**
1. **Newton + tail inversion + Junike Thm 2.1** (shortlist #2): 200 → 15–40 µs now, 2–5 µs after the kernel work.
2. **SIMD cdf kernel for the grid and Newton iterations**: 2049 grid evals ~7 ms → 0.3–0.8 ms; batch threshold can drop from 64 to ~10–20. The hoisted sin/t kernel must be re-verified against the oracle (only the cos kernel was benchmarked end-to-end). CONFIRMED mechanism, cdf variant unverified.
3. **Piecewise-Chebyshev CDF surrogate with panel-bracketed Newton (Olver–Townsend architecture)** for repeated same-(α,β) work and sampling: build 0.5–2 ms (200–600 pdf calls; 5–15x more for α≲0.75 or |β|→1), evaluation 0.1–0.5 µs/pt at ~1e-8–1e-11 with *unclamped* analytic tails; amortized scalar quantiles ~0.2–0.5 µs (400–1000x) after ~200–1500 calls. **[PARTLY_CONFIRMED — architecture verified from the paper; per-eval ns figures are operation-count estimates; the paper's own MATLAB numbers are 15–67 µs/sample with bisection]**. Do *not* use the paper's Möbius map for tails (creates an algebraic endpoint singularity); keep the central-interval/tail-series split. HChebInterp.jl is young/single-maintainer; fallback is ~300 LOC over FastChebInterp.jl. https://arxiv.org/pdf/1307.1223

**Rejected (with measurements).**
- **COS sine-series quantiles**: measured *regression* — 155–314 µs/quantile at ~1e-5 accuracy (N=4096, α=1.5); N scales as ε^{−1/α} because heavy tails force the truncation range; Float64-grade accuracy would need N~1e7. Junike himself states Gil-Pelaez-style inversion (the package's current architecture) is more efficient for stable laws. The *error-propagation theorem* survives and powers recommendation 1.
- **AAA / lightning rational surrogate of Q(p)**: works only after the bounded transform Q(p)·p^{1/α_L}(1−p)^{1/α_R} (raw fit stagnates at relative error 0.81; even one-sided at ~1e-5): degree 53 gives 2.5e-11 uniform relative error, eval 120–600 ns — but build is 30–50 ms vs the Chebyshev route's 0.5–2 ms for the same eval speed. Dominated. Useless for MLE (per-(α,β) refit ×500 evals ≈ +15–25 s). **[PARTLY_CONFIRMED]** https://arxiv.org/abs/2305.03677; https://arxiv.org/abs/2302.02743
- **BL-contour Newton**: 30–80 µs — loses to Newton on the existing kernel everywhere the package works.
- **PINV (Derflinger–Hörmann–Leydold)**: setup 6–48 ms (4k–14k pdf evals), documented heavy-tail failure below u-resolution ~1e-12; dominated by the Chebyshev surrogate. Fallback only.
- **NUFFT/COS batch quantile series**: absolute-error cosine tails go negative; O(N^−α) wall.
- **Quantile mechanics in momentum space (Shaw–McCabe)**: finite convergence radius, no error control; at most a Newton initializer — and the tail-series seed already fills that role.

### 2.4 Coverage and sampling

**Coverage: α∈(0.9,1.1), β≠0 (pdf DomainError at `stable_pdf.jl:19`; cdf worse — all α<1.1 unsupported for β≠0).**
- **Key verified finding: this is a phase-*evaluation* problem, not a quadrature problem.** With the cancellation-stable rewrite (x−ζ)t + ζ·t·expm1((α−1)ln t) and the exact α=1 log-phase limit, plain QuadGK (already a dependency) marches smoothly through α=1: agreement 6.7e-9 with the exact α=1 limit at α=1±1e-6 (2e-9 at ±1e-9), 52–116 µs/pt for |x|≤3, degrading to ~1.8 ms near x~30 before the tail series takes over. Recommended as a correctness branch: days of effort, coverage not speed; eventually unlocks relaxing `fit.jl`'s α≥1.11 clamp. BL curved contours (3.8–10 µs/pt in-band) are the later upgrade but need genuine extra machinery below |α−1|~0.01.
- **Bug found during verification (fix regardless):** at exactly α=1.0, β≠0 the FFT cf in `stable_pdf_fft.jl` silently degenerates to the *symmetric* cf (`tan(π/2)·(|t|^0−1)` evaluates as finite·0 = 0 instead of the (2/π)ln|σt| term) — silent corruption in paths 2/4, narrow band (|α−1|≲1e-10 plus the point 1.0). Fix: expm1 rewrite + explicit α=1 branch.
- **α-snapping (SciPy's fix, band 0.005):** proven robustness precedent (their pre-fix pdf error at α=1.025 was 7.9 orders of magnitude), but blind 0.005 snapping creates a ±0.005 likelihood plateau in α (~1/5 of the α standard error at n=2500–5000) that zeroes FD gradients — if snapping is used at all, keep the band ≤1e-6 and rely on the stable phase rewrite instead. https://github.com/scipy/scipy/issues/12658
- **Asymmetric cdf, α∈[0.5,0.9]:** port Ament–O'Neil's published 181-node F rule from qastable (the repo is already a qastable port — confirm license before vendoring). Accuracy ceiling ~1e-8 *by construction* (t=0 sin/t singularity; A&O say better needs quad-precision rule generation) at ~3–7 µs/pt; test at rtol ~1e-7 and document. https://arxiv.org/abs/1607.04247
- **Certified tail/near-zero cdf bounds (Saenko):** the repo's tail thresholds are already remainder-driven at fixed N=80/90; the marginal win is tolerance-driven N, a *cdf-specific* tail bound (`stable_sym_cdf.jl` currently reuses the pdf bound with an apologetic comment), the near-zero branches, and a second oracle. **[PARTLY_CONFIRMED — Float64 cancellation caps the certified radii]**

**Sampling.**
- **SIMD-batched CMS** (measured locally): 59.6 → 18.5 ns/draw (3.2x) with bulk Xoshiro `rand!`/`randexp!` (2.0 ns/draw of that) + `@turbo` transform; ~5x vs the package's reported ~100 ns. Low risk, same dependency story as shortlist #1. Recommended as a `rand!(d, x)` batch path.
- **Chebyshev inverse-transform sampling** rides on the §2.3 surrogate for free once built.
- **Generalized ziggurat (Zest)**: plausibly fastest *scalar* draws (~99% first-try accept), per-(α,β) table ~1 ms from the fast pdf — **[UNVERIFIED: raw finding only, no local benchmark, no Julia implementation]**. Only worth it if scalar-draw latency is a real workload.
- **PINV sampling**: dominated (see §2.3).

---

## 3. What recent harmonic analysis actually changed since our 2017-era numerics

The package's design (Ament–O'Neil-style precomputed generalized-Gauss rules + asymptotic tail series + uniform FFT for batches) is 2016–2018 state of the art. Since then:

1. **The FFT accuracy floor became a theorem, not an engineering annoyance.** Junike (Numer. Math. 2024) proved that any uniform-grid Fourier-series density recovery for a smooth density with Pareto-index-α tails converges at exactly O(N^−α) — sharp, demonstrated on an α-stable model. The practical corollary: no grid refinement, weight upgrade (Simpson), filter, or basis tweak (COS/SWIFT/PROJ) breaks the ~1e-4-class ceiling; only *analytic* tail handling does. The companion 2022/2025 results supply closed-form truncation/term-count formulas and a rigorous cdf→quantile error-propagation theorem that applies directly to the package's existing quadrature cdf. This retroactively validates the repo's hybrid architecture and tells you precisely which knob (the tail splice) buys accuracy.

2. **NUFFT became commodity infrastructure.** FINUFFT (SISC 2019; manually SIMD-vectorized spreader in v2.3, 2024; Apple Silicon jll binaries) makes "user-chosen quadrature in t, scattered targets in x" a one-call operation. Type-3 turns cf inversion into true quadrature — periodization and interpolation errors vanish *by construction* rather than by tuning; our prototype hit 1.6e-11 at cost parity with the current 1e-4 path. Type-2 precedents in Julia (Le Floc'h 2025, NFFT.jl) confirm the pattern is production-ready in this ecosystem.

3. **Oscillatory and contour quadrature matured from expert craft toward robustness** — Levin-TSVD with no low-frequency breakdown (2022–2024), sinh/conic-contour acceleration applied specifically to stable laws (Boyarchenko–Levendorskii 2018–2020) — but for *this* package their verified value is offline oracles and corner coverage (α≈1, small α), not runtime speed: the fixed-node tables were already near-optimal per region, and full automation (PathFinder) still excludes branch-point phases like t^α.

4. **Rational approximation industrialized.** AAA (2018) → continuum AAA with pole-free guarantees (2024) → tapered lightning achieving Stahl-optimal rates for algebraic singularities (2021–2023), and Boost 1.87 (Dec 2024) shipping four fixed-(α,β) stable-family members as piecewise rationals at 4-eps — existence proof that fixed-parameter stable pdf/cdf/quantile reduce to constant-time formula evaluation. For this package rationals lose to Chebyshev-plus-analytic-tails on build cost, but the Boost precedent justifies the whole surrogate strategy.

5. **Estimation moved into the cf domain.** Matsui–Sueishi (2022, preprint) show a projected-score estimator entirely in cf space reaches ~Cramér–Rao efficiency for α∈[1.0,1.6] with no density inversion at all — sidestepping every Fourier-inversion pathology (kink, tails, aliasing) rather than accelerating it.

6. **The largest verified win here is not from the literature at all.** Loop-invariant hoisting plus portable SIMD transcendentals (SLEEF-class) on the *existing* 2018-style rule measured 7–20x on this machine — bigger than any algorithm swap we tested at equal accuracy. The 2015–2026 harmonic-analysis literature's main contribution to Stable.jl is (a) certifying what the FFT path can never do (item 1), (b) supplying the accuracy-first batch alternative when it matters (item 2), and (c) confirming that the region-switched hybrid the package already uses — now echoed by SciPy 1.9 (2022) and AUB-HTP (2026) — is the correct architecture.
