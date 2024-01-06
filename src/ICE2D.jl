using Distributed, Dates, ArgParse
using Plots, Random, ImageFiltering, Statistics
using Dierckx, Contour, JLD2, DelimitedFiles, JLSO
using Pkg
Pkg.update("Plots")

println("Starting at ", now())

argsettings = ArgParseSettings()
@add_arg_table! argsettings begin
    "--nubeamfilename", "--f"
        help = "The hdf5 filename of nubeam data"
        arg_type = String
        default = "fnb_3235.h5"
    "--magneticfield", "--B0"
        help = "The magnetic field density, B0 [T]"
        arg_type = Float64
        default = 2.09
    "--electrondensity", "--ne"
        help = "The electron number density, n0 [m^-3]"
        arg_type = Float64
        default = 4.71e19
    "--nbidensityfraction", "--nb_ne"
        help = "The fraction of nbi wrt electrons"
        arg_type = Float64
        default = 0.0467
    "--electrontemperatureev", "--te"
        help = "The electron temperature in eV"
        arg_type = Float64
        default = 1.73e3
    "--backgroundprotontemperatureev", "--th"
        help = "The background proton temperature in eV"
        arg_type = Float64
        default = 1.0e3
    "--backgrounddeuterontemperatureev", "--td"
        help = "The background dueteron temperature in eV"
        arg_type = Float64
        default = 1.0e3
    "--ratiothermalprotontoelectrons", "--np_nd"
        help = "The ratio of number densities of thermal H to electrons, n_p / n_e"
        arg_type = Float64
        default = 0.0
    "--nbimassinprotons", "--mn"
        help = "The nbi mass number in units of proton mass"
        arg_type = Int
        default = 2
    "--nringbeams", "--nrbs"
        help = "Number of ring beams to fit with"
        arg_type = Int
        default = 128
    "--timelimithours", "--tlh"
        help = "The time limit to calculate the fit"
        arg_type = Float64
        default = 0.0
    "--niters", "-i"
        help = "The number of restarted fit calculations"
        arg_type = Int
        default = 0
    "--injenergymultiplier"
        help = "Factor to multiply injection energy by"
        arg_type = Float64
        default = 1.0
    "--vthfracinj"
        help = "The ringbeam species will have a temperature equal to vthfracinj of the injection energy"
        arg_type = Float64
        default = 0.1
    "--syntheticspectrumfreqmax"
        help = "The upper limit in frequency of the synthetic spectrum, Hz"
        arg_type = Float64
        default = 200e6
    "--syntheticspectrumnbins"
        help = "The number of bins of the synthetic spectrum"
        arg_type = Integer
        default = 256
    "--initialguessfilepath"
        help = "The path to the results .jlso file to use for the starting state for optimisation"
        arg_type = Union{Nothing,String}
        default = nothing
end

include("NBI.jl")
using .NBI

const parsedargs = parse_args(ARGS, argsettings)
@show parsedargs

const nringbeams = parsedargs["nringbeams"]
const injenergymultiplier = parsedargs["injenergymultiplier"]
const vthfracinj = parsedargs["vthfracinj"]

const name_extension = filter(x -> !isspace(x), length(ARGS) > 0 ? "$(ARGS)" : "$(now())")
const filecontents = [i for i in readlines(open(@__FILE__))]

using LinearMaxwellVlasov

const nbimassinprotons = parsedargs["nbimassinprotons"]
const n0 = parsedargs["electrondensity"]
const B0 = parsedargs["magneticfield"]

@info "Reading nbi data"

const nbidata = try
  _nbidata = NBI.NBIDataEnergyPitch(parsedargs["nubeamfilename"], nbimassinprotons;
    pitchlowcutoff=-1.0, pitchhighcutoff=1.0,
    energykevlowcutoff=0.0, energykevhighcutoff=Inf,
    padpitch_neumann_bc=true)
  @info "Creating an NBIDataEnergyPitch from the data"
  _nbidata
catch err
  _nbidata = NBI.NBIDataVparaVperp(parsedargs["nubeamfilename"], nbimassinprotons;
    cutoffbelowvpara=0.0, cutoffwidthvpara=1e6)
  @info "Creating an NBIDataVparaVperp from the data"
  _nbidata
end


const timelimithours = parsedargs["timelimithours"]
const niters = parsedargs["niters"]

const teev = parsedargs["electrontemperatureev"]
const tpev = parsedargs["backgroundprotontemperatureev"]
const tdev = parsedargs["backgrounddeuterontemperatureev"]
const nd_ne = parsedargs["ratiothermalprotontoelectrons"]
const syntheticspectrumfreqmax = parsedargs["syntheticspectrumfreqmax"]
const syntheticspectrumnbins = parsedargs["syntheticspectrumnbins"]

const ξ = parsedargs["nbidensityfraction"]# * ξfraction
const mn = md = nbimassinprotons * 1836 * LinearMaxwellVlasov.mₑ
const nn = n0 * ξ # number density of nbi
const nd = nd_ne * n0 # number density of background thermal D
const np = n0 - nn - nd # number density of background thermal H
@assert n0 ≈ nn + nd + np

const Ωn = cyclotronfrequency(B0, mn, 1)
const Πn = plasmafrequency(nn, mn, 1)

#@info "Creating nbi interpolator"
#const nbi_interp = NBI.NBIInterpVelocitySpace(nbidata)

#@info "Creating nbi coupled species"
#const _nbi_coupled = NBI.couplednbispecies(nbi_interp, Πn, Ωn)

const initialguessfilepath = parsedargs["initialguessfilepath"]

@info "Calculating nbi fit species with $nringbeams sub-populations"
for _ in 1:niters # iterate and saves result after each
  NBI.differentialevolutionfitspecies(nbidata, Πn, Ωn, nn,
    nringbeams=nringbeams, timelimithours=timelimithours, targetfitness=0.01,
    initialguessfilepath=initialguessfilepath
    )::Vector{NBI.TSpecies}
end
# read it from a file as a const
const _nbi_ringbeamsfit = NBI.differentialevolutionfitspecies(nbidata, Πn, Ωn, nn,
    nringbeams=nringbeams, timelimithours=timelimithours, targetfitness=0.01,
    performoptimisation=false)::Vector{NBI.TSpecies}

@info "nbi_ringbeamsfit has been obtained"

const peakenergykev = NBI.energyofpeakkev(nbidata)
const pitchofmax = NBI.pitchofpeak(nbidata)

const nprocsadded = div(Sys.CPU_THREADS, 2)
addprocs(nprocsadded, exeflags=["--project", "-t 1"])

@everywhere using ProgressMeter # for some reason must be up here on its own
@everywhere using StaticArrays
@everywhere using FastClosures
@everywhere using SpecialFunctions
@everywhere using HDF5
@everywhere using Plots
@everywhere using Dierckx
@everywhere using Interpolations
@everywhere using BlackBoxOptim
@everywhere using JLD2
@everywhere using Serialization
@everywhere using InteractiveUtils
@everywhere using Roots
@everywhere include("NBI.jl")
@everywhere using .NBI # have to re-add this for some reason?
@everywhere begin
  using LinearMaxwellVlasov, LinearAlgebra, WindingNelderMead

  injenergymultiplier = Float64(@fetchfrom 1 injenergymultiplier)
  vthfracinj = Float64(@fetchfrom 1 vthfracinj)

  n0 = Float64(@fetchfrom 1 n0)
  nn = Float64(@fetchfrom 1 nn)
  nd = Float64(@fetchfrom 1 nd)
  np = Float64(@fetchfrom 1 np)
  mn = Float64(@fetchfrom 1 mn)
  B0 = Float64(@fetchfrom 1 B0)
  Πn = Float64(@fetchfrom 1 Πn)
  Ωn = Float64(@fetchfrom 1 Ωn)
  teev = Float64(@fetchfrom 1 teev)
  tpev = Float64(@fetchfrom 1 tpev)
  tdev = Float64(@fetchfrom 1 tdev)

  injectionenergykev = Float64(@fetchfrom 1 peakenergykev)
  peakpitch = Float64(@fetchfrom 1 pitchofmax)

#  nbi_coupled = @fetchfrom 1 _nbi_coupled
  nbi_ringbeamsfit = Vector{NBI.TSpecies}(@fetchfrom 1 _nbi_ringbeamsfit)

  mₑ = LinearMaxwellVlasov.mₑ
  mp = 1836*mₑ
  md = 2*mp

  name_extension = String(@fetchfrom 1 name_extension)

  @assert n0 ≈ nn + nd + np
  Va = sqrt(B0^2/LinearMaxwellVlasov.μ₀/(nd * md + np * mp + nn * mn))

  Ωe = cyclotronfrequency(B0, mₑ, -1)
  Ωd = cyclotronfrequency(B0, md, 1)
  #Ωn = cyclotronfrequency(B0, mn, 1)
  Ωp = cyclotronfrequency(B0, mp, 1)
  Πe = plasmafrequency(n0, mₑ, -1)
  Πd = plasmafrequency(nd, md, 1)
  Πn = plasmafrequency(nn, mn, 1)
  Πh = plasmafrequency(np, mp, 1)
  vthe = thermalspeed(teev, mₑ)
  vthd = thermalspeed(tdev, md)
  vthp = thermalspeed(tpev, mp)

  q₀ = LinearMaxwellVlasov.q₀

  vinj0 = sqrt(2 * injectionenergykev * 1000 * q₀ / mn)
  vinj = sqrt(2 * injectionenergykev * injenergymultiplier * 1000 * q₀ / mn)

  electron_cold = ColdSpecies(Πe, Ωe)
  electron_warm = WarmSpecies(Πe, Ωe, vthe)
  electron_maxw = MaxwellianSpecies(Πe, Ωe, vthe, vthe)

  deuteron_cold = ColdSpecies(Πd, Ωd)
  deuteron_warm = WarmSpecies(Πd, Ωd, vthd)
  deuteron_maxw = MaxwellianSpecies(Πd, Ωd, vthd, vthd)

  proton_maxw = MaxwellianSpecies(Πh, Ωp, vthp, vthp)

  nbi_cold = ColdSpecies(Πn, Ωn)
  nbi_ringbeam = SeparableVelocitySpecies(Πn, Ωn,
    FBeam(vinj * vthfracinj, vinj * peakpitch),
    FRing(vinj * vthfracinj, vinj * sqrt(1-peakpitch^2)))

#  pth = 0.1
#  pinj = 0.7 # TODO make this configurable
#  vcliff = vinj / 10
#  pinj = 0.7
#  function ftanh(vz::T, v⊥::U) where {T, U}
#    v = sqrt(vz^2 + v⊥^2)
#    p = vz / v
#    return v * (1 - tanh((v - vinj) / vcliff)) * exp(-((p - pinj) / pth)^2)
#  end
#  ftanh(vz⊥) = nbi.interp_vz⊥(vz⊥[1], vz⊥[2])
#
#  vzinj = vinj * pinj
#  v⊥inj = vinj * sqrt(1 - pinj^2)
#  ftanhvz⊥ = FCoupledVelocityNumerical(ftanh, (vzinj, v⊥inj), 0.0, vinj*1.1, autonormalise=true)
#  nbi_tanh = CoupledVelocitySpecies(Πn, Ωn, ftanhvz⊥)

#  nbi_coupled = CoupledVelocitySpecies(Πn, Ωn,
#    FBeam(vinj * vthfracinj, vinj * peakpitch),
#    FRing(vinj * vthfracinj, vinj * sqrt(1-peakpitch^2)))
  commonspecies = [electron_maxw]
  !iszero(Πd) && push!(commonspecies, deuteron_maxw)
  !iszero(Πh) && push!(commonspecies, proton_maxw)

  Smms = Plasma([commonspecies..., nbi_ringbeamsfit...])
  Smmc = Plasma([commonspecies..., nbi_cold])
# Smmr = Plasma([commonspecies..., nbi_ringbeam])
# Smmo = Plasma([commonspecies..., nbi_coupled])
# Smmt = Plasma([commonspecies..., nbi_tanh])

  w0 = abs(Ωd)
  k0 = w0 / abs(Va)

  grmax = abs(Ωn) * 0.5
  grmin = -grmax / 4
  function bounds(ω0)
    lb = @SArray [ω0 * 0.5, grmin]
    ub = @SArray [ω0 * 1.2, grmax]
    return (lb, ub)
  end

  options = Options(memoiseparallel=false, memoiseperpendicular=true,
    quadrature_rtol=1e-10, summation_rtol=1e-8)

  function solve_given_ks(K, objective!, coldobjective!)
    ωfA0 = fastzerobetamagnetoacousticfrequency(Va, K, Ωd)
    lb_fA, ub_fA = [ωfA0 / 4], [ωfA0 * 4]

    config = Configuration(K, options)
    function boundedunitcoldobjective!(x::T) where {T}
      output = real(coldobjective!(config, scaleup(lb_fA, ub_fA, x)))
      return maybeaddinf(output, !isinunitbounds(x))
    end
    ω0 = try
      ic = normalise(lb_fA, ub_fA, ωfA0)
      unitresult = Roots.find_zero(x->boundedunitcoldobjective!((@SArray [x])), ic, atol=1e-1)
      scaleup(lb_fA, ub_f, unitresult)
    catch
      ωfA0
    end

    lb, ub = bounds(ω0)

    function boundify(f::T) where {T}
      @inbounds isinbounds(x) = all(i->0 <= x[i] <= 1, eachindex(x))
      maybeaddinf(x::U, addinf::Bool) where {U} = addinf ? x + U(Inf) : x
      bounded(x) = maybeaddinf(f(x), !isinbounds(x))
      return bounded
    end

    config = Configuration(K, options)

    ics = ((@SArray [ω0*0.8, grmax*0.8]),
           (@SArray [ω0*1.1, grmax*0.8]),
           (@SArray [ω0*0.95, grmax*0.4]),
           (@SArray [ω0*0.95, grmax*0.2]),
           (@SArray [ω0*0.95, grmax*0.1]))

    function unitobjective!(c, x::T) where {T}
      return objective!(c,
        T([x[i] * (ub[i] - lb[i]) + lb[i] for i in eachindex(x)]))
    end
    unitobjectivex! = x -> unitobjective!(config, x)
    boundedunitobjective! = boundify(unitobjectivex!)
    xtol_abs = w0 .* (@SArray [1e-4, 1e-5]) ./ (ub .- lb)
    @elapsed for ic ∈ ics
      @assert all(i->lb[i] <= ic[i] <= ub[i], eachindex(ic))
      neldermeadsol = WindingNelderMead.optimise(
        boundedunitobjective!, SArray((ic .- lb) ./ (ub .- lb)),
        1.0e-2 * (@SArray ones(2)); stopval=1e-15, timelimit=3600,
        maxiters=150, ftol_rel=0, ftol_abs=0, xtol_rel=0, xtol_abs=xtol_abs)
      simplex, windingnumber, returncode, numiterations = neldermeadsol
      if (windingnumber == 1 && returncode == :XTOL_REACHED)# || returncode == :STOPVAL_REACHED
        c = deepcopy(config)
        minimiser = if windingnumber == 0
          WindingNelderMead.position(WindingNelderMead.bestvertex(simplex))
        else
          WindingNelderMead.centre(simplex)
        end
        unitobjective!(c, minimiser)
        return c
      end
    end
    return nothing
  end

  function f2Dω!(config::Configuration, x::AbstractArray, plasma, cache)
    config.frequency = Complex(x[1], x[2])
    return det(tensor(plasma, config, cache))
  end
  function f1Dω!(config::Configuration, x, plasma, cache)
    config.frequency = Complex(x[1], 0.0)
    return det(tensor(coldplasma, config, cache))
  end


  function findsolutions(plasma, coldplasma)
    ngridpoints = 2^9
    kzs = range(-1.0, stop=1.0, length=ngridpoints) * k0
    k⊥s = range(0.0, stop=5.0, length=ngridpoints) * k0

    # change order for better distributed scheduling
    k⊥s = shuffle(vcat([k⊥s[i:nprocs():end] for i ∈ 1:nprocs()]...))
    solutions = @sync @showprogress @distributed (vcat) for k⊥ ∈ k⊥s
      cache = Cache()
      objective! = @closure (C, x) -> f2Dω!(C, x, plasma, cache)
      coldobjective! = @closure (C, x) -> f1Dω!(C, x, coldplasma, Cache())
      innersolutions = Vector()
      for (ikz, kz) ∈ enumerate(kzs)
        K = Wavenumber(parallel=kz, perpendicular=k⊥)
        output = solve_given_ks(K, objective!, coldobjective!)
        isnothing(output) && continue
        push!(innersolutions, output)
      end
      innersolutions
    end
    return solutions
  end
end #@everywhere

"Select the largest growth rates if multiple solutions found for a wavenumber"
function selectlargeestgrowthrate(sols)
    imagfreq(s) = imag(s.frequency)
    d = Dict{Any, Vector{eltype(sols)}}()
    for sol in sols
      if haskey(d, sol.wavenumber)
        push!(d[sol.wavenumber], sol)
      else
        d[sol.wavenumber] = [sol]
      end
    end
    output = Vector()
    sizehint!(output, length(sols))
    for (_, ss) in d
      push!(output, ss[findmax(map(imagfreq, ss))[2]])
    end
    return output
end

function selectpropagationrange(sols, lowangle=0, highangle=180)
  function propangle(s)
    kz = para(s.wavenumber)
    k⊥ = perp(s.wavenumber)
    θ = atan.(k⊥, kz) * 180 / pi
  end
  output = Vector()
  for s in sols
    (lowangle <= propangle(s) <= highangle) || continue
    push!(output, s)
  end
  return output
end

Plots.gr()
function plotit(sols, file_extension=name_extension, fontsize=9)
  sols = sort(sols, by=s->imag(s.frequency))
  ωs = [sol.frequency for sol in sols]./w0
  kzs = [para(sol.wavenumber) for sol in sols]./k0
  k⊥s = [perp(sol.wavenumber) for sol in sols]./k0
  xk⊥s = sort(unique(k⊥s))
  ykzs = sort(unique(kzs))

  function make2d(z1d)
    z2d = Array{Union{Float64, Missing}, 2}(zeros(Missing, length(ykzs),
                                            length(xk⊥s)))
    for (j, k⊥) in enumerate(xk⊥s), (i, kz) in enumerate(ykzs)
      index = findlast((k⊥ .== k⊥s) .& (kz .== kzs))
      isnothing(index) || (z2d[i, j] = z1d[index])
    end
    return z2d
  end

  ks = [abs(sol.wavenumber) for sol in sols]./k0
  kθs = atan.(k⊥s, kzs)
  extremaangles = collect(extrema(kθs))

  _median(patch) = median(filter(x->!ismissing(x), patch))
  realωssmooth = make2d(real.(ωs))
  try
    realωssmooth = mapwindow(_median, realωssmooth, (5, 5))
    realωssmooth = imfilter(realωssmooth, Kernel.gaussian(3))
  catch
    @warn "Smoothing failed"
  end

  realspline = nothing
  imagspline = nothing
  try
    smoothing = length(ωs) * 1e-4
    realspline = Dierckx.Spline2D(xk⊥s, ykzs, realωssmooth'; kx=4, ky=4, s=smoothing)
    imagspline = Dierckx.Spline2D(k⊥s, kzs, imag.(ωs); kx=4, ky=4, s=smoothing)
  catch err
    @warn "Caught $err. Continuing."
  end

  function plotangles(;writeangles=true)
    for θdeg ∈ vcat(collect.((30:5:80, 81:99, 100:5:150))...)
      θ = θdeg * π / 180
      xs = sin(θ) .* [0, maximum(ks)]
      ys = cos(θ) .* [0, maximum(ks)]
      linestyle = mod(θdeg, 5) == 0 ? :solid : :dash
      Plots.plot!(xs, ys, linecolor=:grey, linewidth=0.5, linestyle=linestyle)
      writeangles || continue
      if atan(maximum(k⊥s), maximum(kzs)) < θ < atan(maximum(k⊥s), minimum(kzs))
        xi, yi = xs[end], ys[end]
        isapprox(yi, maximum(kzs), rtol=0.01, atol=0.01) && (yi += 0.075)
        isapprox(xi, maximum(k⊥s), rtol=0.01, atol=0.01) && (xi += 0.1)
        isapprox(xi, minimum(k⊥s), rtol=0.01, atol=0.01) && (xi -= 0.2)
        Plots.annotate!([(xi, yi, text("\$ $(θdeg)^{\\circ}\$", fontsize, :black))])
      end
    end
  end
  function plotcontours(spline, contourlevels, skipannotation=x->false)
    isnothing(spline) && return nothing
    x, y = sort(unique(k⊥s)), sort(unique(kzs))
    z = evalgrid(spline, x, y)
    for cl ∈ Contour.levels(Contour.contours(x, y, z, contourlevels))
      lvl = try; Int(Contour.level(cl)); catch; Contour.level(cl); end
      for line ∈ Contour.lines(cl)
          xs, ys = Contour.coordinates(line)
          θs = atan.(xs, ys)
          ps = sortperm(θs, rev=true)
          xs, ys, θs = xs[ps], ys[ps], θs[ps]
          mask = minimum(extremaangles) .< θs .< maximum(extremaangles)
          any(mask) || continue
          xs, ys = xs[mask], ys[mask]
          Plots.plot!(xs, ys, color=:grey, linewidth=0.5)
          skipannotation(ys) && continue
          yi, index = findmax(ys)
          xi = xs[index]
          if !(isapprox(xi, minimum(k⊥s), rtol=0.01, atol=0.5) ||
               isapprox(yi, maximum(kzs), rtol=0.01, atol=0.01))
            continue
          end
          isapprox(xi, maximum(k⊥s), rtol=0.1, atol=0.5) && continue
          isapprox(yi, maximum(kzs), rtol=0.1, atol=0.5) && (yi += 0.075)
          isapprox(xi, minimum(k⊥s), rtol=0.1, atol=0.5) && (xi = -0.1)
          Plots.annotate!([(xi, yi, text("\$\\it{$lvl}\$", fontsize-1, :black))])
      end
    end
  end

  msize = 2
  mshape = :square
  function plotter2d(z, xlabel, ylabel, colorgrad,
      climmin=minimum(z[@. !ismissing(z)]), climmax=maximum(z[@. !ismissing(z)]))
    zcolor = make2d(z)
    dx = (xk⊥s[2] - xk⊥s[1]) / (length(xk⊥s) - 1)
    dy = (ykzs[2] - ykzs[1]) / (length(ykzs) - 1)
    h = Plots.heatmap(xk⊥s, ykzs, zcolor, framestyle=:box, c=colorgrad,
      xlims=(minimum(xk⊥s) - dx/2, maximum(xk⊥s) + dx/2),
      ylims=(minimum(ykzs) - dy/2, maximum(ykzs) + dy/2),
      clims=(climmin, climmax), xticks=0:Int(round(maximum(xk⊥s))),
      xlabel=xlabel, ylabel=ylabel)
  end
  xlabel = "\$\\mathrm{Perpendicular\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
  zs = real.(ωs)
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), 0.0, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  Plots.plot!(legend=false)
  Plots.savefig("ICE2D_real_$file_extension.pdf")

  ω0s = [fastzerobetamagnetoacousticfrequency(Va, sol.wavenumber, Ωd) for
    sol in sols] / w0
  zs = real.(ωs) ./ ω0s
  climmin = minimum(zs)
  climmax = maximum(zs)
  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), climmin, climmax)
  Plots.title!(" ")
  plotangles(writeangles=false)
  Plots.plot!(legend=false)
  Plots.savefig("ICE2D_real_div_guess_$file_extension.pdf")

#  zs = iseven.(Int64.(floor.(real.(ωs))))
#  climmax = maximum(zs)
#  plotter2d(zs, xlabel, ylabel, Plots.cgrad(), 0.0, climmax)
#  Plots.title!(" ")
#  plotangles(writeangles=false)
#  plotcontours(realspline, collect(1:50), y -> y[end] < 0)
#  Plots.plot!(legend=false)
#  Plots.savefig("ICE2D_evenfloorreal_real_$file_extension.pdf")

  zs = imag.(ωs)
  climmax = maximum(zs)
  colorgrad = Plots.cgrad([:cyan, :black, :darkred, :red, :orange, :yellow])
  plotter2d(zs, xlabel, ylabel, colorgrad, -climmax / 4, climmax)
  Plots.title!(" ")
  Plots.plot!(legend=false)
  plotcontours(realspline, collect(1:50), y -> y[end] < 0)
  plotangles(writeangles=false)
  Plots.savefig("ICE2D_imag_$file_extension.pdf")

  zs[zs .< 0] .= NaN
  zs .= log10.(zs)
  climmax = maximum(zs)
  colorgrad = Plots.cgrad([:cyan, :black, :darkred, :red, :orange, :yellow])
  plotter2d(zs, xlabel, ylabel, colorgrad, -climmax / 4, climmax)
  Plots.title!(" ")
  Plots.plot!(legend=false)
  plotcontours(realspline, collect(1:50), y -> y[end] < 0)
  plotangles(writeangles=false)
  Plots.savefig("ICE2D_log10_imag_$file_extension.pdf")


  colorgrad = Plots.cgrad()

  xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
  ylabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"

  imaglolim = 1e-5

  mask = shuffle(findall(@. (imag(ωs) > imaglolim)))
  @warn "Scatter plots rendering with $(length(mask)) points."
  if sum(mask) > 0
    perm = sortperm(imag.(ωs[mask]))
    h0 = Plots.scatter(real.(ωs[mask][perm]), kzs[mask][perm],
      zcolor=imag.(ωs[mask][perm]), framestyle=:box, lims=:round,
      markersize=msize+1, markerstrokewidth=0, markershape=:circle,
      c=colorgrad, yticks=unique(Int.(round.(ykzs))),
      xticks=0:2:Int(round(maximum(real, ωs[mask]))),
      xlims=(0, ceil(maximum(real.(ωs[mask][perm])))+1),
      xlabel=xlabel, ylabel=ylabel, legend=:topleft)
    Plots.plot!(legend=false)
    Plots.savefig("ICE2D_KF_$file_extension.pdf")

    ylabel = "\$\\mathrm{Growth\\ Rate} \\ [\\Omega_{i}]\$"
    h1 = Plots.scatter(real.(ωs[mask]), imag.(ωs[mask]),
      zcolor=kzs[mask], framestyle=:box,
      markersize=msize+1, markerstrokewidth=0, markershape=:circle,
      xlims=(0, ceil(maximum(real.(ωs[mask][perm])))+1),
      c=colorgrad, lims=:round,
      xticks=0:2:Int(round(maximum(real, ωs[mask]))),
      xlabel=xlabel, ylabel=ylabel, legend=:topleft)
    Plots.plot!(legend=false)
    Plots.savefig("ICE2D_GF_$file_extension.pdf")

    ylabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
    xlabel = "\$\\mathrm{Parallel\\ Wavenumber} \\ [\\Omega_{i} / V_A]\$"
    h2 = Plots.scatter(kzs[mask], real.(ωs[mask]),
      zcolor=log10.(imag.(ωs[mask])), framestyle=:box,
      markersize=msize+1, markerstrokewidth=0, markershape=:circle,
      c=colorgrad, xlims=(-2, 2), lims=:round,
      xlabel=xlabel, ylabel=ylabel, legend=:topleft)
    Plots.plot!(legend=false)
    Plots.savefig("ICE2D_WK_$file_extension.pdf")

    xlabel = "\$\\mathrm{Frequency} \\ [\\Omega_{i}]\$"
    ylabel = "\$\\mathrm{Propagation\\ Angle} \\ [^{\\circ}]\$"

    colorgrad = Plots.cgrad([:cyan, :black, :darkred, :red, :orange, :yellow])
    h4 = Plots.scatter(real.(ωs[mask]), kθs[mask] .* 180 / π,
      zcolor=log10.(imag.(ωs[mask])),
      markersize=msize, markerstrokewidth=0, markershape=mshape, framestyle=:box,
      c=Plots.cgrad([:black, :darkred, :red, :orange, :yellow]),
      clims=(0, maximum(log10.(imag.(ωs[mask])))), lims=:round,
      xlims=(0, ceil(maximum(real.(ωs[mask][perm])))+1),
      xticks=0:2:Int(round(maximum(real, ωs[mask]))),
      yticks=(0:10:180), xlabel=xlabel, ylabel=ylabel)
    Plots.plot!(legend=false)
    Plots.savefig("ICE2D_TF_$file_extension.pdf")

    Plots.xlabel!(h1, "")
    Plots.xticks!(h1, 0:-1)
    Plots.plot(h1, h0, link=:x, layout=@layout [a; b])
    Plots.savefig("ICE2D_Combo_$file_extension.pdf")
  end

   freq_bins_Hz = collect(range(0, stop=syntheticspectrumfreqmax, length=syntheticspectrumnbins))
   syntheticspectrum = zeros(syntheticspectrumnbins)
   for ω in ωs
     fHz = real(ω) * w0 / 2π # real frequency in Hz
     index = findlast(x->fHz > x, freq_bins_Hz)
     (1 <= index <= syntheticspectrumnbins) || continue
     γHz = imag(ω) * w0 / 2π # imag frequency in Hz
     syntheticspectrum[index] = max(syntheticspectrum[index], γHz)
   end
   xlabel = "\$\\mathrm{Frequency} \\ [\\mathrm{MHz}]\$"
   ylabel = "\$\\mathrm{Growth\\ Rate} \\ [{\\mathrm{MHz}}]\$"
   Plots.plot(freq_bins_Hz ./ 1e6, syntheticspectrum ./ 1e6, xlabel=xlabel, ylabel=ylabel)
   Plots.savefig("ICE2D_SyntheticSpectrum_$file_extension.pdf")
end

if true#false#
  @time plasmasols = findsolutions(Smms, Smmc)
  plasmasols = selectlargeestgrowthrate(plasmasols)
  @show length(plasmasols)
  @time plotit(plasmasols)
  @save "solutions2D_$name_extension.jld" filecontents plasmasols w0 k0
  rmprocs(nprocsadded)
else
  rmprocs(nprocsadded)
  @load "solutions2D_$name_extension.jld" filecontents plasmasols w0 k0
  @time plotit(plasmasols)
end

println("Ending at ", now())


