# Hoisted, SIMD-batched evaluation of the fixed-node Fourier quadratures.
#
# After scaling the nodes, the (M)-parameterization inversion integrals
#
#   pdf(x) = (1/π) ∫₀^∞ cos((x-ζ)t + ζ t^α) e^(-t^α) dt
#   cdf(x) = 1/2 + (1/π) ∫₀^∞ sin((x-ζ)t + ζ t^α) e^(-t^α) dt/t
#
# collapse to Σⱼ wⱼ·cos(x·tⱼ + uⱼ) (resp. 0.5 + Σⱼ wⱼ·sin(x·tⱼ + uⱼ)) with
# x-independent tⱼ = r·gxⱼ, uⱼ = ζ(tⱼ^α - tⱼ), wⱼ = (r/π)·gwⱼ·e^(-tⱼ^α) (/tⱼ
# for the cdf), r the rank scaling. StableQuadKernel caches those vectors per
# (α, β), so evaluating one point costs one cos/sin and two fma per node and
# the point loop vectorizes with LoopVectorization.
#
# The identity cos(-x·t - u) = cos(x·t + u) (and sin's sign flip, which
# realizes F(x; β) = 1 - F(-x; -β)) makes the kernels side-agnostic: the
# reflection f(x; α, β) = f(-x; α, -β) is built into the algebra, so a single
# kernel covers the whole window between the two tail-series thresholds.
# Verified: kernel values match the region-switched scalar dispatchers to
# machine precision inside that window on both sides of ζ.

struct StableQuadKernel
    α::Float64
    β::Float64
    ζ::Float64
    # pdf quadrature window (ζ-relative thresholds where the tail series takes over)
    pdf_upper_x::Float64
    pdf_lower_x::Float64
    pdf_n_inf::Int
    # cdf quadrature window; NaN bounds when the cdf quadrature is unsupported
    cdf_upper_x::Float64
    cdf_lower_x::Float64
    cdf_n_inf::Int
    t_pdf::Vector{Float64}
    u_pdf::Vector{Float64}
    w_pdf::Vector{Float64}
    t_cdf::Vector{Float64}
    u_cdf::Vector{Float64}
    w_cdf::Vector{Float64}
    # x-independent tail-series coefficients (upper = β side, lower = -β side),
    # so a tail point costs one evalpoly instead of ~90 loggamma/sin calls
    tail_pdf_upper::Vector{Float64}
    tail_pdf_lower::Vector{Float64}
    tail_cdf_upper::Vector{Float64}
    tail_cdf_lower::Vector{Float64}
end

# Smallest x - ζ beyond which the truncated tail series reaches machine
# precision. Evaluated in log space: the direct product overflows through
# gamma(a * n_inf) ≈ 1e136 once |ζ| ≳ 53 (α within ~0.01|β| of 1), which
# would silently disable tail routing.
function tail_series_threshold(a::Float64, z::Float64, n_inf::Int)
    log_c = (n_inf / 2) * log1p(z^2) + log(a) - log(pi * eps(Float64)) +
            loggamma(a * n_inf) - loggamma(Float64(n_inf))
    return exp(log_c / (a * n_inf - 1)) + z
end

pdf_series_terms(a::Float64, b::Float64) = b == 0.0 ? 42 : (a >= 1.1 ? 80 : 90)

# x-independent coefficients of the tail series: with y = (x - ζ)^(-α),
#   pdf tail  = (α/π) (x-ζ)^(-α-1) Σₖ cₖ yᵏ
#   ccdf tail = (1/π) (x-ζ)^(-α)   Σₖ c'ₖ yᵏ
# matching stable_pdf_series_infinity / stable_ccdf_series_infinity exactly.
function tail_series_coefficients(a::Float64, b::Float64, max_coef::Int; cdf::Bool)
    zeta = -b * tan(pi * a / 2)
    angle = pi / 2 * a - atan(zeta)
    sqrt_1_plus_zeta2 = sqrt(1 + zeta^2)
    c = Vector{Float64}(undef, max_coef + 1)
    term_sign = 1.0
    geometric_part = 1.0
    for k in 0:max_coef
        geometric_part *= sqrt_1_plus_zeta2
        log_factorial = cdf ? loggamma(k + 2.0) : loggamma(k + 1.0)
        c[k + 1] = term_sign * exp(loggamma(a * (k + 1)) - log_factorial) *
                   geometric_part * sin(angle * (k + 1))
        term_sign = -term_sign
    end
    return c
end

function tail_pdf_eval(c::Vector{Float64}, x::Float64, a::Float64, zeta::Float64)
    y = (x - zeta)^(-a)
    return a / pi * (x - zeta)^(-a - 1) * evalpoly(y, c)
end

function tail_ccdf_eval(c::Vector{Float64}, x::Float64, a::Float64, zeta::Float64)
    y = (x - zeta)^(-a)
    return y / pi * evalpoly(y, c)
end

function scaled_kernel_vectors(gx::Vector{Float64}, gw::Vector{Float64},
                               a::Float64, z::Float64; cdf::Bool)
    r = (-log(eps(Float64)))^(1 / a)
    t = r .* gx
    ta = t .^ a
    u = z .* (ta .- t)
    w = (r / pi) .* gw .* exp.(-ta)
    cdf && (w ./= t)
    return t, u, w
end

function StableQuadKernel(a::Float64, b::Float64)
    -1.0 <= b <= 1.0 || throw(DomainError(b, "skewness β must satisfy -1 ≤ β ≤ 1"))
    z = -b * tan(a * pi / 2)

    if b == 0.0
        0.5 <= a <= 2.0 ||
            throw(DomainError(a, "symmetric stable quadrature requires α ∈ [0.5, 2]"))
        gx_p, gw_p = GX_SYM_PDF, GW_SYM_PDF
        gx_c, gw_c = GX_SYM_CDF, GW_SYM_CDF
        n_inf_p = n_inf_c = 42
    else
        if a >= 1.1
            gx_p, gw_p = GX_PDF_86, GW_PDF_86
            n_inf_p = 80
        elseif 0.5 <= a <= 0.9
            gx_p, gw_p = GX_PDF_94, GW_PDF_94
            n_inf_p = 90
        else
            throw(DomainError(a,
                "asymmetric (β ≠ 0) stable pdf requires α ∈ [0.5, 0.9] ∪ [1.1, 2]"))
        end
        # the asymmetric cdf quadrature only exists for α ≥ 1.1
        gx_c, gw_c = a >= 1.1 ? (GX_CDF, GW_CDF) : (Float64[], Float64[])
        n_inf_c = 80
    end

    t_p, u_p, w_p = scaled_kernel_vectors(gx_p, gw_p, a, z; cdf = false)
    pdf_up = tail_series_threshold(a, z, n_inf_p)
    pdf_lo = -tail_series_threshold(a, -z, n_inf_p)
    tail_p_u = tail_series_coefficients(a, b, n_inf_p; cdf = false)
    tail_p_l = b == 0.0 ? tail_p_u : tail_series_coefficients(a, -b, n_inf_p; cdf = false)

    if isempty(gx_c)
        t_c, u_c, w_c = Float64[], Float64[], Float64[]
        cdf_up, cdf_lo = NaN, NaN
        tail_c_u, tail_c_l = Float64[], Float64[]
    else
        t_c, u_c, w_c = scaled_kernel_vectors(gx_c, gw_c, a, z; cdf = true)
        cdf_up = tail_series_threshold(a, z, n_inf_c)
        cdf_lo = -tail_series_threshold(a, -z, n_inf_c)
        tail_c_u = tail_series_coefficients(a, b, n_inf_c; cdf = true)
        tail_c_l = b == 0.0 ? tail_c_u : tail_series_coefficients(a, -b, n_inf_c; cdf = true)
    end

    return StableQuadKernel(a, b, z, pdf_up, pdf_lo, n_inf_p, cdf_up, cdf_lo, n_inf_c,
                            t_p, u_p, w_p, t_c, u_c, w_c,
                            tail_p_u, tail_p_l, tail_c_u, tail_c_l)
end

has_cdf_kernel(k::StableQuadKernel) = !isempty(k.t_cdf)

#### scalar evaluation on a prebuilt kernel

function kernel_pdf(k::StableQuadKernel, x::Float64)
    if x > k.pdf_upper_x
        return tail_pdf_eval(k.tail_pdf_upper, x, k.α, k.ζ)
    elseif x < k.pdf_lower_x
        return tail_pdf_eval(k.tail_pdf_lower, -x, k.α, -k.ζ)
    end
    t, u, w = k.t_pdf, k.u_pdf, k.w_pdf
    acc = 0.0
    @inbounds @simd for j in eachindex(t, u, w)
        acc += w[j] * cos(x * t[j] + u[j])
    end
    return acc
end

function kernel_cdf(k::StableQuadKernel, x::Float64)
    has_cdf_kernel(k) || throw(DomainError(k.α,
        "asymmetric (β ≠ 0) stable cdf requires α ∈ [1.1, 2]"))
    if x > k.cdf_upper_x
        return 1.0 - tail_ccdf_eval(k.tail_cdf_upper, x, k.α, k.ζ)
    elseif x < k.cdf_lower_x
        # reflected upper-tail series, taken directly for full relative accuracy
        return tail_ccdf_eval(k.tail_cdf_lower, -x, k.α, -k.ζ)
    end
    t, u, w = k.t_cdf, k.u_cdf, k.w_cdf
    acc = 0.5
    @inbounds @simd for j in eachindex(t, u, w)
        acc += w[j] * sin(x * t[j] + u[j])
    end
    return acc
end

#### batched evaluation

# out[i] = Σⱼ w[j]·cos(xs[i]·t[j] + u[j]) — vectorized over points and nodes
function kernel_sum_cos!(out::Vector{Float64}, t::Vector{Float64}, u::Vector{Float64},
                         w::Vector{Float64}, xs::Vector{Float64})
    @tturbo for i in eachindex(xs)
        acc = 0.0
        for j in eachindex(t)
            acc += w[j] * cos(xs[i] * t[j] + u[j])
        end
        out[i] = acc
    end
    return out
end

function kernel_sum_sin!(out::Vector{Float64}, t::Vector{Float64}, u::Vector{Float64},
                         w::Vector{Float64}, xs::Vector{Float64})
    @tturbo for i in eachindex(xs)
        acc = 0.0
        for j in eachindex(t)
            acc += w[j] * sin(xs[i] * t[j] + u[j])
        end
        out[i] = acc
    end
    return out
end

# Evaluate the standard-law pdf (density = true) or cdf at every point of `xs`,
# routing each point to the SIMD quadrature kernel inside the window and to
# the tail series outside it.
function kernel_batch(k::StableQuadKernel, xs::Vector{Float64}; density::Bool)
    density || has_cdf_kernel(k) || throw(DomainError(k.α,
        "asymmetric (β ≠ 0) stable cdf requires α ∈ [1.1, 2]"))
    upper = density ? k.pdf_upper_x : k.cdf_upper_x
    lower = density ? k.pdf_lower_x : k.cdf_lower_x

    out = Vector{Float64}(undef, length(xs))
    body_idx = Int[]
    body_x = Float64[]
    for (i, x) in pairs(xs)
        if x > upper
            out[i] = density ? tail_pdf_eval(k.tail_pdf_upper, x, k.α, k.ζ) :
                     1.0 - tail_ccdf_eval(k.tail_cdf_upper, x, k.α, k.ζ)
        elseif x < lower
            out[i] = density ? tail_pdf_eval(k.tail_pdf_lower, -x, k.α, -k.ζ) :
                     tail_ccdf_eval(k.tail_cdf_lower, -x, k.α, -k.ζ)
        else
            push!(body_idx, i)
            push!(body_x, x)
        end
    end

    if !isempty(body_idx)
        body_out = Vector{Float64}(undef, length(body_x))
        if density
            kernel_sum_cos!(body_out, k.t_pdf, k.u_pdf, k.w_pdf, body_x)
        else
            kernel_sum_sin!(body_out, k.t_cdf, k.u_cdf, k.w_cdf, body_x)
            body_out .+= 0.5
        end
        out[body_idx] .= body_out
    end
    return out
end
