# Parameter estimation: McCulloch's quantile estimator for a fast consistent
# starting point, then maximum likelihood via JuMP + Ipopt with the FFT-based
# log-likelihood.

"""
    fitcullstable(x) -> [μ, α, β, σ]

McCulloch's (1986) quantile-based estimator of the stable parameters, in the
(M)/S0 parameterization used by [`AlphaStable`](@ref). Fast and consistent but
less efficient than MLE; used as the starting point (and fallback) for
[`fitmlestable`](@ref).
"""
function fitcullstable(x::AbstractVector{<:Real})
    length(x) >= 5 ||
        throw(ArgumentError("McCulloch estimation requires at least 5 observations"))

    x05 = percentile(x, 5)
    x25 = percentile(x, 25)
    x50 = percentile(x, 50)
    x75 = percentile(x, 75)
    x95 = percentile(x, 95)
    x75 > x25 ||
        throw(ArgumentError("cannot estimate stable parameters from (nearly) constant data"))

    # quantile statistics
    va = (x95 - x05) / (x75 - x25)
    vb = (x95 + x05 - 2 * x50) / (x95 - x05)
    vs = x75 - x25

    # McCulloch's interpolation tables
    tva = [2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0]
    tvb = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    ta = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    tb = [0.0, 0.25, 0.5, 0.75, 1.0]

    psi1 = [2.000 2.000 2.000 2.000 2.000 2.000 2.000;
            1.916 1.924 1.924 1.924 1.924 1.924 1.924;
            1.808 1.813 1.829 1.829 1.829 1.829 1.829;
            1.729 1.730 1.737 1.745 1.745 1.745 1.745;
            1.664 1.663 1.663 1.668 1.676 1.676 1.676;
            1.563 1.560 1.553 1.548 1.547 1.547 1.547;
            1.484 1.480 1.471 1.460 1.448 1.438 1.438;
            1.391 1.386 1.378 1.364 1.337 1.318 1.318;
            1.279 1.273 1.266 1.250 1.210 1.184 1.150;
            1.128 1.121 1.114 1.101 1.067 1.027 0.973;
            1.029 1.021 1.014 1.004 0.974 0.935 0.874;
            0.896 0.892 0.887 0.883 0.855 0.823 0.769;
            0.818 0.812 0.806 0.801 0.780 0.756 0.691;
            0.698 0.695 0.692 0.689 0.676 0.656 0.595;
            0.593 0.590 0.588 0.586 0.579 0.563 0.513]

    psi2 = [0.000 2.160 1.000 1.000 1.000 1.000 1.000;
            0.000 1.592 3.390 1.000 1.000 1.000 1.000;
            0.000 0.759 1.800 1.000 1.000 1.000 1.000;
            0.000 0.482 1.048 1.694 1.000 1.000 1.000;
            0.000 0.360 0.760 1.232 2.229 1.000 1.000;
            0.000 0.253 0.518 0.823 1.575 1.000 1.000;
            0.000 0.203 0.410 0.632 1.244 1.906 1.000;
            0.000 0.165 0.332 0.499 0.943 1.560 1.000;
            0.000 0.136 0.271 0.404 0.689 1.230 2.195;
            0.000 0.109 0.216 0.323 0.539 0.827 1.917;
            0.000 0.096 0.190 0.284 0.472 0.693 1.759;
            0.000 0.082 0.163 0.243 0.412 0.601 1.596;
            0.000 0.074 0.147 0.220 0.377 0.546 1.482;
            0.000 0.064 0.128 0.191 0.330 0.478 1.362;
            0.000 0.056 0.112 0.167 0.285 0.428 1.274]

    psi3 = [1.908 1.908 1.908 1.908 1.908;
            1.914 1.915 1.916 1.918 1.921;
            1.921 1.922 1.927 1.936 1.947;
            1.927 1.930 1.943 1.961 1.987;
            1.933 1.940 1.962 1.997 2.043;
            1.939 1.952 1.988 2.045 2.116;
            1.946 1.967 2.022 2.106 2.211;
            1.955 1.984 2.067 2.188 2.333;
            1.965 2.007 2.125 2.294 2.491;
            1.980 2.040 2.205 2.435 2.696;
            2.000 2.085 2.311 2.624 2.973;
            2.040 2.149 2.461 2.886 3.356;
            2.098 2.244 2.676 3.265 3.912;
            2.189 2.392 3.004 3.844 4.775;
            2.337 2.635 3.542 4.808 6.247;
            2.588 3.073 4.534 6.636 9.144]

    psi4 = [0.0 0.0 0.0 0.0 0.0;
            0.0 -0.017 -0.032 -0.049 -0.064;
            0.0 -0.030 -0.061 -0.092 -0.123;
            0.0 -0.043 -0.088 -0.132 -0.179;
            0.0 -0.056 -0.111 -0.170 -0.232;
            0.0 -0.066 -0.134 -0.206 -0.283;
            0.0 -0.075 -0.154 -0.241 -0.335;
            0.0 -0.084 -0.173 -0.276 -0.390;
            0.0 -0.090 -0.192 -0.310 -0.447;
            0.0 -0.095 -0.208 -0.346 -0.508;
            0.0 -0.098 -0.223 -0.383 -0.576;
            0.0 -0.099 -0.237 -0.424 -0.652;
            0.0 -0.096 -0.250 -0.469 -0.742;
            0.0 -0.089 -0.262 -0.520 -0.853;
            0.0 -0.078 -0.272 -0.581 -0.997;
            0.0 -0.061 -0.279 -0.659 -1.198]

    # compute estimates by interpolating through the tables
    tvai1 = maximum(push!(findall(tva .<= va), 1))
    tvai2 = minimum(push!(findall(tva .>= va), 15))
    tvbi1 = maximum(push!(findall(tvb .<= abs(vb)), 1))
    tvbi2 = minimum(push!(findall(tvb .>= abs(vb)), 7))

    dista = tva[tvai2] - tva[tvai1]
    if dista != 0
        dista = (va - tva[tvai1]) / dista
    end
    distb = tvb[tvbi2] - tvb[tvbi1]
    if distb != 0
        distb = (abs(vb) - tvb[tvbi1]) / distb
    end

    psi1b1 = dista * psi1[tvai2, tvbi1] + (1 - dista) * psi1[tvai1, tvbi1]
    psi1b2 = dista * psi1[tvai2, tvbi2] + (1 - dista) * psi1[tvai1, tvbi2]
    alpha = distb * psi1b2 + (1 - distb) * psi1b1
    psi2b1 = dista * psi2[tvai2, tvbi1] + (1 - dista) * psi2[tvai1, tvbi1]
    psi2b2 = dista * psi2[tvai2, tvbi2] + (1 - dista) * psi2[tvai1, tvbi2]
    beta = sign(vb) * (distb * psi2b2 + (1 - distb) * psi2b1)

    tai1 = maximum(push!(findall(ta .>= alpha), 1))
    tai2 = minimum(push!(findall(ta .<= alpha), 16))
    tbi1 = maximum(push!(findall(tb .<= abs(beta)), 1))
    tbi2 = minimum(push!(findall(tb .>= abs(beta)), 5))

    dista = ta[tai2] - ta[tai1]
    if dista != 0
        dista = (alpha - ta[tai1]) / dista
    end
    distb = tb[tbi2] - tb[tbi1]
    if distb != 0
        distb = (abs(beta) - tb[tbi1]) / distb
    end

    psi3b1 = dista * psi3[tai2, tbi1] + (1 - dista) * psi3[tai1, tbi1]
    psi3b2 = dista * psi3[tai2, tbi2] + (1 - dista) * psi3[tai1, tbi2]
    sigma = vs / (distb * psi3b2 + (1 - distb) * psi3b1)
    psi4b1 = dista * psi4[tai2, tbi1] + (1 - dista) * psi4[tai1, tbi1]
    psi4b2 = dista * psi4[tai2, tbi2] + (1 - dista) * psi4[tai1, tbi2]
    zeta = sign(beta) * sigma * (distb * psi4b2 + (1 - distb) * psi4b1) + x50

    # McCulloch's ζ is exactly the location parameter of the S0/(M)
    # parameterization used by this package (his δ refers to S1)
    mu = zeta

    # correct estimates for out-of-range values
    alpha = clamp(alpha, eps(Float64), 2.0)
    sigma = max(sigma, eps(Float64))
    beta = clamp(beta, -1.0, 1.0)

    return [mu, alpha, beta, sigma]
end

# Negative log-likelihood used for MLE, based on the SIMD-batched quadrature
# kernel (exact to quadrature accuracy — the clamped α ∈ [1.1, 2] region is
# fully covered by the 86-node rule). Parameters are clamped to the
# numerically supported region, and |β| is damped near α = 2 where it is
# barely identifiable.
function negloglike(p::AbstractVector{<:Real}, data::AbstractVector{Float64})
    μ = Float64(p[1])
    α = clamp(Float64(p[2]), 1.1, 2.0)
    β = clamp(Float64(p[3]), -1.0, 1.0)
    σ = max(Float64(p[4]), 1e-9)
    beta_penalty = (α > 1.818 && abs(β) > 0.5) ? length(data) * (abs(β) - 0.5)^2 : 0.0
    d = AlphaStable(μ, α, β, σ; check_args = false)
    # floor each contribution (matching the old FFT objective's √eps pdf floor)
    # so the objective and its FD gradient stay finite at |β| → 1 trial points,
    # where light-tail quadrature noise can make logpdf_batch return -Inf
    log_floor = log(sqrt(eps(Float64)))
    return -sum(Base.Fix2(max, log_floor), logpdf_batch(d, data)) + beta_penalty
end

"""
    fitmlestable(x; x0 = fitcullstable(x)) -> (params, loglike, status)

Maximum-likelihood estimate of the stable parameters of the data `x`
(`NaN`s are dropped), maximizing the FFT-based log-likelihood with Ipopt.
Returns a `NamedTuple` with the parameter vector `[μ, α, β, σ]`, the attained
log-likelihood and the Ipopt termination status. Falls back to the starting
point if the solver fails to improve on it. The stability index is restricted
to `α ∈ [1.11, 2)` (the asymmetric cdf/pdf machinery needs `α ≥ 1.1`).
"""
function fitmlestable(indata::AbstractVector{<:Real};
                      x0::Union{Nothing, AbstractVector{<:Real}} = nothing)
    data = collect(Float64, filter(!isnan, indata))
    length(data) >= 5 ||
        throw(ArgumentError("MLE fitting requires at least 5 non-NaN observations"))
    start = x0 === nothing ? fitcullstable(data) : collect(Float64, x0)

    lb = [minimum(data), 1.11, -1.0, min(1e-6, start[4] / 10)]
    ub = [maximum(data), 2.0 - 1e-3, 1.0, 10.0 * max(start[4], 1e-6)]
    p0 = clamp.(start, lb, ub)

    nll(μ, α, β, σ) = negloglike([μ, α, β, σ], data)
    # finite-difference steps relative to each parameter's natural scale: μ and
    # σ live on the data scale, α and β on O(1). An absolute floor of 1e-4
    # (the old behavior) makes the gradient so noisy for small-scale data
    # (e.g. return series with σ ~ 0.02) that Ipopt's line search thrashes,
    # costing ~100 objective evaluations per iteration instead of ~10.
    fd_scale = (max(start[4], 1e-12), 1.0, 1.0, max(start[4], 1e-12))
    function nll_gradient(g::AbstractVector{T}, μ::T, α::T, β::T, σ::T) where {T}
        x = [μ, α, β, σ]
        for i in eachindex(x)
            h = 1e-4 * max(fd_scale[i], abs(x[i]))
            xp = copy(x)
            xm = copy(x)
            xp[i] += h
            xm[i] -= h
            g[i] = (negloglike(xp, data) - negloglike(xm, data)) / (2 * h)
        end
        return nothing
    end

    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_attribute(model, "sb", "yes")
    set_attribute(model, "max_iter", 150)
    # tolerances matched to the finite-difference gradient noise — demanding
    # more just makes Ipopt run to the iteration limit
    set_attribute(model, "tol", 1e-4)
    set_attribute(model, "acceptable_tol", 1e-2)
    set_attribute(model, "acceptable_iter", 8)
    # gradients are finite differences; no exact Hessian is available
    set_attribute(model, "hessian_approximation", "limited-memory")

    @variable(model, lb[i] <= p[i = 1:4] <= ub[i], start = p0[i])
    @operator(model, op_nll, 4, nll, nll_gradient)
    @objective(model, Min, op_nll(p[1], p[2], p[3], p[4]))
    optimize!(model)

    phat = primal_status(model) in (FEASIBLE_POINT, NEARLY_FEASIBLE_POINT) ?
           value.(p) : copy(p0)
    # never return something worse than the starting point
    if negloglike(phat, data) > negloglike(p0, data)
        phat = copy(p0)
    end
    phat[2] = clamp(phat[2], 1.1001, 2.0)
    phat[3] = clamp(phat[3], -1.0, 1.0)
    phat[4] = max(phat[4], 1e-12)

    return (params = phat, loglike = -negloglike(phat, data),
            status = termination_status(model))
end

function fit_mle(::Type{<:AlphaStable}, x::AbstractArray{<:Real})
    fit = fitmlestable(vec(x))
    p = fit.params
    return AlphaStable(p[1], p[2], p[3], p[4])
end

# legacy form taking a distribution instance as a dummy first argument
fit_mle(::AlphaStable, x::AbstractArray{<:Real}) = fit_mle(AlphaStable, x)

"""
    fitstable(x) -> AlphaStable

Convenience wrapper for `fit_mle(AlphaStable, x)`.
"""
fitstable(x::AbstractArray{<:Real}) = fit_mle(AlphaStable, x)

"""
    refitstable(d::AlphaStable, x) -> AlphaStable

Re-run the MLE on `x` warm-started from the parameters of `d`.
"""
function refitstable(d::AlphaStable, x::AbstractArray{<:Real})
    fit = fitmlestable(vec(x); x0 = collect(Float64.(params(d))))
    p = fit.params
    return AlphaStable(p[1], p[2], p[3], p[4])
end
