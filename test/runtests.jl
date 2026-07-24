using Test
using Stable
using Distributions
using QuadGK
using Random
using SpecialFunctions
using Statistics

@testset "Stable.jl" begin

    @testset "construction and parameters" begin
        d = AlphaStable()
        @test params(d) == (0.0, 2.0, 0.0, 1.0)
        @test AlphaStable(1.5) isa AlphaStable{Float64}
        @test params(AlphaStable(1.5, 0.3)) == (0.0, 1.5, 0.3, 1.0)
        @test AlphaStable(0, 2, 0, 1) isa AlphaStable{Float64}
        @test AlphaStable(0, 1.5, 0, 2) isa AlphaStable{Float64}

        @test_throws DomainError AlphaStable(0.0, 1.5, 0.0, 0.0)   # σ must be > 0
        @test_throws DomainError AlphaStable(0.0, 0.0, 0.0, 1.0)   # α must be > 0
        @test_throws DomainError AlphaStable(0.0, 2.5, 0.0, 1.0)   # α must be ≤ 2
        @test_throws DomainError AlphaStable(0.0, 1.5, 1.5, 1.0)   # |β| must be ≤ 1
        @test AlphaStable(0.0, 2.5, 0.0, 1.0; check_args = false) isa AlphaStable

        d32 = AlphaStable(0.0f0, 1.5f0, 0.0f0, 1.0f0)
        @test partype(d32) === Float32
        @test convert(AlphaStable{Float64}, d32) isa AlphaStable{Float64}
        @test convert(AlphaStable{Float32}, d32) === d32
        @test isfinite(pdf(d32, 0.5f0))
        @test pdf(d32, 0.5f0) ≈ pdf(AlphaStable(0.0, 1.5, 0.0, 1.0), 0.5)

        d = AlphaStable(0.5, 1.5, 0.3, 2.0)
        @test location(d) == 0.5
        @test scale(d) == 2.0
        @test insupport(d, 1e10) && insupport(d, -1e10)
        @test minimum(d) == -Inf && maximum(d) == Inf
    end

    @testset "closed forms: α = 2 (Gaussian) and α = 1 (Cauchy)" begin
        μ, σ = 0.4, 1.3
        dg = AlphaStable(μ, 2.0, 0.0, σ)
        ng = Normal(μ, sqrt(2) * σ)
        dc = AlphaStable(μ, 1.0, 0.0, σ)
        nc = Cauchy(μ, σ)
        for x in -6.0:0.7:6.0
            @test pdf(dg, x) ≈ pdf(ng, x) atol = 1e-14
            @test cdf(dg, x) ≈ cdf(ng, x) atol = 1e-14
            @test pdf(dc, x) ≈ pdf(nc, x) atol = 1e-14
            @test cdf(dc, x) ≈ cdf(nc, x) atol = 1e-14
        end
        # the special-cased branches must agree with the α → 2 / α → 1 numerics
        @test Stable.stable_pdf(0.3, 2.0, 0.0) ≈ Stable.stable_pdf(0.3, 1.999999, 0.0) atol = 1e-6
        @test Stable.stable_cdf(0.7, 2.0, 0.0) ≈ Stable.stable_cdf(0.7, 1.999999, 0.0) atol = 1e-6
        @test Stable.stable_pdf(0.3, 1.0, 0.0) ≈ Stable.stable_pdf(0.3, 1.000001, 0.0) atol = 1e-5
    end

    @testset "pdf against adaptive-quadrature oracle" begin
        for (a, b) in [(1.5, 0.5), (1.8, -0.7), (1.2, 0.9), (0.7, 0.3), (1.6, 0.0), (0.6, -1.0)]
            for x in -8.0:0.37:8.0
                p_tab = Stable.stable_pdf(x, a, b)
                p_ref = Stable.stdstable_pdf_quad(x, a, b)
                @test p_tab >= 0
                if p_ref > 1e-12
                    @test p_tab ≈ p_ref rtol = 1e-8
                end
            end
        end
        # σ scaling: f(x; μ, σ) = f((x-μ)/σ; 0, 1)/σ
        d = AlphaStable(0.7, 1.4, -0.5, 2.5)
        @test pdf(d, 1.9) ≈ Stable.stable_pdf((1.9 - 0.7) / 2.5, 1.4, -0.5) / 2.5
        # normalization
        for (a, b) in [(1.5, 0.5), (0.7, 0.3)]
            total, _ = quadgk(x -> Stable.stable_pdf(x, a, b),
                              -Inf, -20.0, -2.0, 0.0, 2.0, 20.0, Inf; rtol = 1e-8)
            @test total ≈ 1.0 atol = 1e-6
        end
    end

    @testset "cdf" begin
        for (a, b) in [(1.5, 0.0), (1.5, 0.5), (1.2, -0.8), (1.9, 0.3)]
            f(x) = Stable.stable_pdf(x, a, b)
            for xq in (-3.0, -0.5, 1.0, 4.0)
                num, _ = quadgk(f, -Inf, -20.0, xq; rtol = 1e-9)
                @test Stable.stable_cdf(xq, a, b) ≈ num atol = 1e-8
            end
            # monotone
            grid = -10.0:0.5:10.0
            cs = [Stable.stable_cdf(x, a, b) for x in grid]
            @test issorted(cs)
        end
        # deep-tail complement identities (regression: symmetric left tail)
        @test Stable.stable_cdf(-9.0, 1.5, 0.0) + Stable.stable_cdf(9.0, 1.5, 0.0) ≈ 1.0 atol = 1e-13
        @test Stable.stable_cdf(-9.0, 1.5, 0.0) > 0.005
        @test Stable.stable_cdf(-30.0, 1.2, 0.0) + Stable.stable_cdf(30.0, 1.2, 0.0) ≈ 1.0 atol = 1e-13
        # asymmetric reflection F(x; β) = 1 - F(-x; -β)
        for x in (-12.0, -3.0, 0.4, 7.0)
            @test Stable.stable_cdf(x, 1.4, 0.6) ≈ 1 - Stable.stable_cdf(-x, 1.4, -0.6) atol = 1e-13
        end
        d = AlphaStable(0.5, 1.5, 0.3, 2.0)
        @test cdf(d, -1e9) < 1e-6
        @test cdf(d, 1e9) > 1 - 1e-6
        # tail probabilities carry full relative accuracy: the deep lower cdf
        # and the upper ccdf are the series sums themselves, and tail decay
        # follows the α power law
        @test cdf(d, -1e6) > 0
        @test cdf(d, -1e7) / cdf(d, -1e6) ≈ 10.0^(-1.5) rtol = 1e-3
        @test ccdf(d, 1e7) / ccdf(d, 1e6) ≈ 10.0^(-1.5) rtol = 1e-3
        dsym = AlphaStable(0.0, 1.3, 0.0, 1.0)
        for x in (50.0, 1e4, 1e8)
            @test ccdf(dsym, x) ≈ cdf(dsym, -x) rtol = 1e-12   # exact symmetry
        end
        @test ccdf(AlphaStable(0.0, 1.0, 0.0, 1.0), 1e8) ≈ 1 / (pi * 1e8) rtol = 1e-6
        @test ccdf(AlphaStable(0.0, 2.0, 0.0, 1.0), 10.0) ≈ erfc(5.0) / 2 rtol = 1e-13
        @test ccdf(d, 0.5) + cdf(d, 0.5) ≈ 1.0 atol = 1e-14
    end

    @testset "quantile" begin
        for (a, b) in [(1.5, 0.5), (1.9, -0.2), (1.15, 0.9)]
            d = AlphaStable(0.5, a, b, 2.0)
            for p in (1e-6, 1e-3, 0.05, 0.5, 0.95, 0.999, 1 - 1e-6)
                @test cdf(d, quantile(d, p)) ≈ p atol = 1e-8
            end
        end
        d = AlphaStable(0.0, 1.6, 0.3, 1.0)
        @test quantile(d, 0.0) == -Inf
        @test quantile(d, 1.0) == Inf
        @test_throws DomainError quantile(d, 1.5)
        # extreme probabilities: the tail-series-seeded Newton solver is exact
        # (no clamping on the scalar path)
        for p in (1e-12, 1e-9, 1e-4, 1 - 1e-9, 1 - 1e-12)
            q = quantile(d, p)
            @test isfinite(q)
            @test cdf(d, q) ≈ p rtol = 1e-8
        end
        # subnormal p gets a log-space seed (regression: NaN via Inf - Inf)
        @test isfinite(quantile(d, 5e-324))
        @test cdf(d, quantile(d, 1e-300)) ≈ 1e-300 rtol = 1e-6
        # a quantile beyond floatmax is ±Inf, not NaN
        @test quantile(AlphaStable(0.0, 0.7, 0.0, 1.0), 1e-300) == -Inf
        # closed-form branches are complement-free (regression: -Inf / tan
        # saturation for p ≲ 1e-16)
        dg = AlphaStable(0.0, 2.0, 0.0, 1.0)
        for p in (1e-17, 1e-100, 1e-300)
            @test isfinite(quantile(dg, p))
            @test quantile(dg, p) ≈ quantile(Normal(0.0, sqrt(2)), p) rtol = 1e-12
        end
        dc = AlphaStable(0.0, 1.0, 0.0, 1.0)
        for p in (1e-17, 1e-100, 1e-300)
            @test quantile(dc, p) ≈ -1 / (pi * p) rtol = 1e-10
        end
        # batch fast path (≥ 64 points goes through the interpolated grid)
        ps = collect(range(0.001, 0.999; length = 1500))
        xq = quantile(d, ps)
        @test maximum(abs(cdf(d, xq[i]) - ps[i]) for i in eachindex(ps)) < 5e-5
        ps100 = collect(range(0.01, 0.99; length = 100))
        xq100 = quantile(d, ps100)
        @test maximum(abs(cdf(d, xq100[i]) - ps100[i]) for i in eachindex(ps100)) < 5e-5
        # small batches use the scalar path
        @test quantile(d, [0.25, 0.75]) ≈ [quantile(d, 0.25), quantile(d, 0.75)]
    end

    @testset "statistics" begin
        # in the (M) parameterization E[X] = μ + σζ for α > 1
        for (a, b) in [(1.7, 0.6), (1.3, -0.9)]
            d = AlphaStable(1.0, a, b, 2.0)
            num, _ = quadgk(x -> x * pdf(d, x), -Inf, Inf; rtol = 1e-8)
            @test mean(d) ≈ num atol = 1e-3
        end
        @test mean(AlphaStable(0.7, 2.0, 0.0, 1.0)) ≈ 0.7 atol = 1e-14
        @test isnan(mean(AlphaStable(0.0, 1.0, 0.0, 1.0)))
        @test var(AlphaStable(0.0, 2.0, 0.0, 1.5)) == 2 * 1.5^2
        @test var(AlphaStable(0.0, 1.9, 0.0, 1.5)) == Inf
        @test median(AlphaStable(0.4, 1.5, 0.0, 2.0)) == 0.4
        d = AlphaStable(0.4, 1.5, 0.3, 2.0)
        @test median(d) ≈ quantile(d, 0.5)
        @test mode(AlphaStable(0.4, 1.5, 0.0, 2.0)) == 0.4
    end

    @testset "characteristic function" begin
        d = AlphaStable(0.3, 1.6, 0.4, 1.2)
        @test cf(d, 0.0) == complex(1.0)
        for t in (-2.0, -0.5, 0.7, 3.0)
            @test abs(cf(d, t)) ≈ exp(-(1.2 * abs(t))^1.6) rtol = 1e-12
            # cf must be the Fourier transform of the pdf (S0 consistency)
            num, _ = quadgk(x -> cis(t * x) * pdf(d, x), -Inf, Inf; rtol = 1e-8)
            @test cf(d, t) ≈ num atol = 1e-6
        end
        # α = 1 branch
        d1 = AlphaStable(0.0, 1.0, 0.0, 2.0)
        @test cf(d1, 0.5) ≈ exp(-2.0 * 0.5) atol = 1e-12
        # α → 1 continuity with skew: the ((σ|t|)^(1-α) - 1) factor must be
        # evaluated via expm1 — the direct pow-then-subtract cancels
        # catastrophically for |α - 1| ≲ 1e-10 and the skewness term degrades
        # toward the symmetric cf
        dskew = AlphaStable(0.3, 1.0, 0.5, 1.2)
        for a in (1 - 1e-12, 1 + 1e-12), t in (-2.0, -0.5, 0.7, 3.0)
            @test cf(AlphaStable(0.3, a, 0.5, 1.2), t) ≈ cf(dskew, t) rtol = 1e-9
        end
    end

    @testset "sampling" begin
        Random.seed!(20260724)
        for (a, b) in [(1.5, 0.5), (1.8, -0.4), (2.0, 0.0), (1.0, 0.0), (1.2, 0.9)]
            d = AlphaStable(0.0, a, b, 1.0)
            n = 20_000
            xs = sort!(rand(d, n))
            ks = maximum(abs(cdf(d, xs[i]) - i / n) for i in 1:n)
            @test ks < 0.015    # 99% KS critical value is ≈ 0.0115
        end
        # α = 2: exact Gaussian with variance 2σ²
        xs = rand(AlphaStable(1.0, 2.0, 0.0, 1.0), 50_000)
        @test mean(xs) ≈ 1.0 atol = 0.05
        @test var(xs) ≈ 2.0 rtol = 0.05
        # reproducible with a seeded RNG
        d = AlphaStable(0.0, 1.5, 0.3, 1.0)
        @test rand(Xoshiro(1), d) == rand(Xoshiro(1), d)
        @test length(rand(d, 7)) == 7
    end

    @testset "batch evaluation (SIMD quadrature kernel)" begin
        for (a, b) in [(1.5, 0.5), (1.2, 0.9), (0.7, 0.3), (1.9, -0.6), (1.5, 0.0),
                       (0.7, 0.0), (1.05, 0.0), (2.0, 0.0), (1.0, 0.0)]
            d = AlphaStable(0.3, a, b, 1.7)
            xs = collect(range(-40, 40; length = 501))
            @test maximum(abs.(pdf_batch(d, xs) .- pdf.(d, xs))) < 1e-13
            if b == 0.0 || a >= 1.1
                @test maximum(abs.(cdf_batch(d, xs) .- cdf.(d, xs))) < 1e-13
            else
                @test_throws DomainError cdf_batch(d, xs)
            end
            lp = logpdf_batch(d, xs)
            @test all(isfinite, lp)
            @test maximum(abs.(lp .- logpdf.(d, xs))) < 1e-10
        end
        @test_throws DomainError pdf_batch(AlphaStable(0.0, 1.0, 0.5, 1.0), [0.3])
        # cdf_batch reports cdf-specific support, matching scalar cdf's message
        @test_throws DomainError cdf_batch(AlphaStable(0.0, 0.7, 0.3, 1.0), [0.3])
        # input genericity
        d = AlphaStable(0.0, 1.5, 0.3, 1.0)
        @test pdf_batch(d, Float64[]) == Float64[]
        @test cdf_batch(d, Int[]) == Float64[]
        @test pdf_batch(d, [-2, 0, 3]) ≈ pdf.(d, [-2.0, 0.0, 3.0])
        @test cdf_batch(d, Float32[-1.5, 0.5]) ≈ cdf.(d, [-1.5, 0.5]) atol = 1e-7
        @test pdf_fft(d, Float64[]) == Float64[]
    end

    @testset "pdf_fft and logpdf_fft" begin
        for (a, b) in [(1.5, 0.5), (1.8, -0.3), (1.2, 0.0)]
            d = AlphaStable(0.3, a, b, 1.7)
            xs = collect(range(-9, 9; length = 1201))
            @test maximum(abs.(pdf_fft(d, xs) .- pdf.(d, xs))) < 5e-6
            @test all(isfinite, logpdf_fft(d, xs))
        end
        # far tails route through the certified series (exact relative accuracy)
        d = AlphaStable(0.0, 1.5, 0.5, 1.0)
        xt = [-500.0, -60.0, 60.0, 500.0]
        @test maximum(abs.(pdf_fft(d, xt) ./ pdf.(d, xt) .- 1)) < 1e-10
        # α ∈ (0.9, 1.1) with β ≠ 0: the quadrature is unsupported but the
        # FFT + series path covers it — check against a stably-rewritten
        # adaptive Fourier inversion (phase s·t + ζ·t·expm1((α-1)·log t))
        a, b = 0.95, 0.5
        z = -b * tan(a * pi / 2)
        function band_oracle(s)
            f(t) = t == 0 ? 1.0 :
                   cos(s * t + z * t * expm1((a - 1) * log(t))) * exp(-t^a)
            int, _ = quadgk(f, 0.0, 300.0; rtol = 1e-12, atol = 1e-15)
            return int / pi
        end
        db = AlphaStable(0.0, a, b, 1.0)
        ss = collect(range(-15, 15; length = 41))
        pf = pdf_fft(db, ss)
        @test maximum(abs(pf[i] - band_oracle(ss[i])) for i in eachindex(ss)) < 1e-5
        # far tails in the α ≈ 1 band route through the series (regression:
        # tail_series_threshold overflowed to Inf, leaving aliased FFT values)
        function band_tail_oracle(s, aa, bb)
            zz = -bb * tan(aa * pi / 2)
            # α = 1 needs its own phase — the expm1 form collapses to the
            # symmetric integrand there
            phase(t) = aa == 1 ? s * t + bb * (2 / pi) * t * log(t) :
                       s * t + zz * t * expm1((aa - 1) * log(t))
            g(t) = t == 0 ? 1.0 : cos(phase(t)) * exp(-t^aa)
            int, _ = quadgk(g, 0.0, 300.0; rtol = 1e-12, atol = 1e-300)
            return int / pi
        end
        for aa in (0.999, 1.001)
            dt = AlphaStable(0.0, aa, 0.5, 1.0)
            @test pdf_fft(dt, [5000.0])[1] ≈ band_tail_oracle(5000.0, aa, 0.5) rtol = 1e-6
        end
        # exactly α = 1 with β ≠ 0: dedicated tail handling (asymptote beyond
        # |s| = 2000, ~1e-3 relative)
        d1 = AlphaStable(0.0, 1.0, 0.5, 1.0)
        @test pdf_fft(d1, [5000.0])[1] ≈ 1.5 / (pi * 5000.0^2) rtol = 1e-2
        @test pdf_fft(d1, [100.0])[1] ≈ band_tail_oracle(100.0, 1.0, 0.5) rtol = 1e-6
        # small α: documented ~1e-6 body accuracy holds down to α = 0.5
        for (aa, bb) in [(0.5, 0.0), (0.7, 0.5)]
            ds = AlphaStable(0.0, aa, bb, 1.0)
            xs_s = collect(range(-6, 6; length = 121))
            @test maximum(abs.(pdf_fft(ds, xs_s) .-
                               [Stable.stable_pdf(x, aa, bb) for x in xs_s])) < 5e-6
        end
    end

    @testset "unsupported parameter regions throw" begin
        @test_throws DomainError pdf(AlphaStable(0.0, 1.0, 0.5, 1.0), 0.3)   # β ≠ 0, α ∈ (0.9, 1.1)
        @test_throws DomainError pdf(AlphaStable(0.0, 0.95, 0.5, 1.0), 0.3)
        @test_throws DomainError cdf(AlphaStable(0.0, 0.8, 0.5, 1.0), 0.3)   # asymmetric cdf needs α ≥ 1.1
        @test_throws DomainError pdf(AlphaStable(0.0, 0.3, 0.0, 1.0), 0.3)   # α < 0.5
        @test isfinite(pdf(AlphaStable(0.0, 0.7, 0.3, 1.0), 0.3))            # 94-node branch works
    end

    @testset "fitting" begin
        Random.seed!(7)
        dtrue = AlphaStable(-1.0, 1.35, -0.6, 0.05)
        data = rand(dtrue, 2500)

        p0 = fitcullstable(data)
        @test abs(p0[1] - (-1.0)) < 0.05      # μ (scale is 0.05, so this is 1σ)
        @test abs(p0[2] - 1.35) < 0.25        # α
        @test abs(p0[3] - (-0.6)) < 0.35      # β
        @test 0.7 < p0[4] / 0.05 < 1.3        # σ

        fit = fitmlestable(data)
        @test abs(fit.params[1] - (-1.0)) < 0.02
        @test abs(fit.params[2] - 1.35) < 0.12
        @test abs(fit.params[3] - (-0.6)) < 0.25
        @test 0.85 < fit.params[4] / 0.05 < 1.15
        @test fit.loglike >= -Stable.negloglike(p0, data) - 1e-6

        dhat = fit_mle(AlphaStable, data)
        @test dhat isa AlphaStable{Float64}
        @test collect(params(dhat)) ≈ fit.params
        @test params(fit_mle(AlphaStable(), data)) == params(dhat)   # legacy instance form
        @test params(fitstable(data)) == params(dhat)

        # NaNs are dropped
        @test fit_mle(AlphaStable, [data; NaN; NaN]) isa AlphaStable

        # warm-started refit should not lose likelihood
        drefit = refitstable(dhat, data)
        @test sum(logpdf_fft(drefit, data)) >= fit.loglike - 1.0

        @test_throws ArgumentError fitcullstable(ones(100))
        @test_throws ArgumentError fitmlestable(Float64[])
    end

    @testset "type coherence across partypes" begin
        # evaluation API returns Float64 for every partype; closed-form
        # statistics return float(T)
        d64 = AlphaStable(0.5, 1.5, 0.3, 2.0)
        d32 = AlphaStable(0.5f0, 1.5f0, 0.3f0, 2.0f0)
        dbig = AlphaStable(big"0.5", big"1.5", big"0.3", big"2.0")
        for d in (d64, d32, dbig)
            @test pdf(d, 0.5) isa Float64
            @test logpdf(d, 0.5) isa Float64
            @test cdf(d, 0.5) isa Float64
            @test ccdf(d, 0.5) isa Float64
            @test quantile(d, 0.25) isa Float64
            @test cf(d, 0.7) isa ComplexF64
            @test rand(Xoshiro(1), d) isa Float64
            @test eltype(pdf_batch(d, [0.5, 1.0])) === Float64
            @test eltype(cdf_batch(d, [0.5, 1.0])) === Float64
            @test eltype(pdf_fft(d, [0.5, 1.0])) === Float64
        end
        @test mean(d32) isa Float32
        @test var(d32) isa Float32
        @test mean(dbig) isa BigFloat
        # evaluation values agree across partypes (all Float64 inside)
        @test pdf(d32, 0.5) ≈ pdf(d64, 0.5) rtol = 1e-6
        @test pdf(dbig, 0.5) == pdf(d64, 0.5)
        # inference of the hot scalar API
        for f in (x -> pdf(x, 0.5), x -> cdf(x, 0.5), x -> quantile(x, 0.25),
                  mean, var, mode)
            @test (@inferred f(d32)) !== nothing
            @test (@inferred f(d64)) !== nothing
        end
        # non-float partypes: NaN/Inf statistics branches must not throw
        # (regression: oftype(::Rational, NaN) is an InexactError)
        dint = AlphaStable{Int}(0, 1, 0, 1)
        @test isnan(mean(dint)) && mean(dint) isa Float64
        @test var(dint) == Inf && var(dint) isa Float64
        drat = AlphaStable{Rational{Int}}(1 // 2, 3 // 2, 3 // 10, 2 // 1)
        @test isnan(mode(drat)) && mode(drat) isa Float64
        @test var(drat) == Inf && var(drat) isa Float64
        @test isfinite(mean(drat)) && mean(drat) isa Float64
        @test pdf(drat, 0.5) isa Float64
    end

    @testset "logpdf" begin
        d = AlphaStable(0.0, 1.5, 0.3, 2.0)
        for x in (-5.0, 0.0, 3.0, 40.0)
            @test logpdf(d, x) ≈ log(pdf(d, x))
        end
        @test isfinite(logpdf(d, 1e6))   # far tail stays finite via the series
    end

end
