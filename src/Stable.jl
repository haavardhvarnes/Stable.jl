# Alpha-stable distribution as an extension of Distributions.jl, using
# Zolotarev's (M) parameterization (= Nolan's S0) throughout.
#
# Main references:
# - Weron: http://sfb649.wiwi.hu-berlin.de/papers/pdf/SFB649DP2005-008.pdf
# - Nolan: http://fs2.american.edu/jpnolan/www/stable/density.pdf
# - Ament & O'Neil: https://arxiv.org/pdf/1607.04247.pdf
#   (numerics ported from https://gitlab.com/s_ament/qastable)
# - Borak, Härdle & Weron: http://prac.im.pwr.edu.pl/~hugo/publ/SFB2005-008_Borak_Haerdle_Weron.pdf
module Stable

using Distributions
using FFTW
using Interpolations
using LoopVectorization
using Optim
using QuadGK
using Random
using SpecialFunctions
using Statistics
using StatsBase

import Distributions: @distr_support, pdf, logpdf, cdf, ccdf, quantile, cf,
    location, scale, params, partype, fit_mle
import Statistics: mean, var, median
import StatsBase: mode
import Random: rand

export AlphaStable
# re-export the extended Distributions/Statistics API
export pdf, logpdf, cdf, ccdf, quantile, cf, mean, var, median, mode,
    location, scale, params, partype, fit_mle
# fitting helpers
export fitcullstable, fitmlestable, fitstable, refitstable
# fast batch evaluation
export pdf_batch, logpdf_batch, cdf_batch, pdf_fft, logpdf_fft

# standard-law numerics (μ = 0, σ = 1)
include("stable_sym_pdf_fourier_integral.jl")
include("stable_sym_pdf.jl")
include("stable_sym_cdf_integral.jl")
include("stable_sym_cdf.jl")
include("stable_pdf_fourier_integral.jl")
include("stable_pdf_series_infinity.jl")
include("stable_pdf_series_zero.jl")
include("stable_pdf.jl")
include("stable_cdf_integral.jl")
include("stable_cdf_series_infinity.jl")
include("stable_cdf.jl")
include("quadrature_kernel.jl")

# reference integrands (documentation / verification)
include("stable_pdf_integrand.jl")
include("stable_pdf_fourier_integrand.jl")
include("stable_cdf_integrand.jl")

# characteristic function, FFT pdf and adaptive-quadrature pdf
include("stable_pdf_fft.jl")

# Distributions.jl interface
include("alphastable.jl")

# parameter estimation
include("fit.jl")

end
