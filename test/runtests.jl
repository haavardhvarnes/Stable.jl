using Test
using Stable
using Distributions
using QuadGK
using Random
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
        # batch fast path (≥ 700 points goes through the interpolated grid)
        ps = collect(range(0.001, 0.999; length = 1500))
        xq = quantile(d, ps)
        @test maximum(abs(cdf(d, xq[i]) - ps[i]) for i in eachindex(ps)) < 5e-5
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

    @testset "pdf_fft and logpdf_fft" begin
        for (a, b) in [(1.5, 0.5), (1.8, -0.3), (1.2, 0.0)]
            d = AlphaStable(0.3, a, b, 1.7)
            xs = collect(range(-9, 9; length = 1201))
            @test maximum(abs.(pdf_fft(d, xs) .- pdf.(d, xs))) < 5e-4
            @test all(isfinite, logpdf_fft(d, xs))
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

    @testset "logpdf" begin
        d = AlphaStable(0.0, 1.5, 0.3, 2.0)
        for x in (-5.0, 0.0, 3.0, 40.0)
            @test logpdf(d, x) ≈ log(pdf(d, x))
        end
        @test isfinite(logpdf(d, 1e6))   # far tail stays finite via the series
    end

end
