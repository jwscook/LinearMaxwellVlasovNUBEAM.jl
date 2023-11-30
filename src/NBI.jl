module NBI

using SpecialFunctions, LinearAlgebra, Random, AutoHashEquals
using BlackBoxOptim, Plots
using LoopVectorization, Base.Threads, Polyester
using LinearMaxwellVlasov
using JLSO, HDF5
using Dierckx, Interpolations

Random.seed!(0)

const ϵ = LinearMaxwellVlasov.ϵ₀
const mₑ = LinearMaxwellVlasov.mₑ
const q₀ = LinearMaxwellVlasov.q₀
const ncomponents_per_ringbeam = 5
const TSpecies = SeparableVelocitySpecies{Float64, Float64, FBeam{Float64, Float64, LinearMaxwellVlasov.ShiftedMaxwellianParallel{Float64, Float64}}, FRing{Float64, Float64, LinearMaxwellVlasov.ShiftedMaxwellianPerpendicular{Float64, Float64, Float64}}}


abstract type AbstractNBIData end

@auto_hash_equals struct NBIDataEnergyPitch <: AbstractNBIData
  massnumber::Int
  fname::String
  fdata::Matrix{Float64}
  pdata::Vector{Float64}
  sqrt1_p²data::Vector{Float64}
  edata::Vector{Float64}
  fmax::Float64
  speedmin::Float64
  speedrange::Float64
end

"""
    NBIDataEnergyPitch(fname,massnumber=2;energysamplestep=1,pitchsamplestep=1,energykevlowcutoff=0.0,energykevhighcutoff=Inf,pitchlowcutoff=-Inf,pitchhighcutoff=Inf,padpitch_neumann_bc=false,nzerocolpad=1)

description

...
# Arguments
- `fname`: the filename of the NUBEAM data
- `massnumber=2`:  mass of the ions, in units of proton masses
Optional args:
- `energysamplestep=1`: sample the data every nth point in the energy direction
- `pitchsamplestep=1`:  sample the data every nth point in the pitch direction
- `energykevlowcutoff=0.0`: cutoff the data below this energy
- `energykevhighcutoff=Inf`:  cutoff the data above this energy
- `pitchlowcutoff=-Inf`: cutoff the data below this pitch
- `pitchhighcutoff=Inf`: cutoff the data above this pitch
- `padpitch_neumann_bc=false`: copy the data out to pitches of -1, 1
- `nzerocolpad=1`: keep this many columns of zeros
...

# Example
```julia
```
"""
function NBIDataEnergyPitch(fname, massnumber=2;
    energysamplestep=1,
    pitchsamplestep=1,
    energykevlowcutoff=0.0,
    energykevhighcutoff=Inf,
    pitchlowcutoff=-Inf,
    pitchhighcutoff=Inf,
    padpitch_neumann_bc=false,
    nzerocolpad=1)

  fdata = h5read(fname, "fnb")
  pdata = h5read(fname, "pdata")[:, 1]
  edata = h5read(fname, "edata")[1, :]

  if padpitch_neumann_bc
    if minimum(pdata) > -1
      prepend!(pdata, -1)
      fdata = vcat(fdata[1, :]', fdata)
    end
    if maximum(pdata) < 1
      push!(pdata, 1)
      fdata = vcat(fdata, fdata[end, :]')
    end
  end

  pitch0 = pdata[1:pitchsamplestep:end]
  energy0 = edata[1:energysamplestep:end]
  fnb0 = fdata[1:pitchsamplestep:end, 1:energysamplestep:end]

  indp1 = findfirst(pitch0 .>= pitchlowcutoff)
  indp2 = findlast(pitch0 .<= pitchhighcutoff)
  isnothing(indp1) && (indp1 = 1)
  isnothing(indp2) && (indp2 = length(pitch0))
  inde1 = findfirst(energy0 .>= energykevlowcutoff)
  inde2 = findlast(energy0 .<= energykevhighcutoff)

  energy = energy0[inde1:inde2]
  pitch = pitch0[indp1:indp2]
  fnb = fnb0[indp1:indp2, inde1:inde2]

  lastnonzerocolumn = findlast(!iszero, eachcol(fnb))
  useuptocolumn = min(lastnonzerocolumn + nzerocolpad, length(energy))
  energy = energy[1:useuptocolumn]
  fnb = fnb[:, 1:useuptocolumn]

  fmax = maximum(fnb[:])
  fnb ./= fmax

  mass = massnumber * 1836mₑ
  v²perkeV = 1000q₀ * 2 / mass

  speedmin = sqrt(minimum(edata) * v²perkeV)
  speedrange = sqrt(maximum(edata) * v²perkeV) - speedmin

  return NBIDataEnergyPitch(massnumber, fname, fnb, pitch, sqrt.(1 .- pitch.^2), energy, fmax, speedmin, speedrange)
end

pitchofpeak(n::NBIDataEnergyPitch) = n.pdata[findmax(n.fdata)[2][1]]
energyofpeakkev(n::NBIDataEnergyPitch) = n.edata[findmax(n.fdata)[2][2]]
maxenergy(n::NBIDataEnergyPitch) = n.edata[end]
maxspeed(n::NBIDataEnergyPitch) = sqrt(maxenergy(n) * 1000q₀ * 2 / (n.massnumber * 1836mₑ))

pitchmin(n::NBIDataEnergyPitch) = minimum(n.pdata)
pitchrange(n::NBIDataEnergyPitch) = maximum(n.pdata) - pitchmin(n)

@auto_hash_equals struct NBIDataVparaVperp <: AbstractNBIData
  massnumber::Int
  fname::String
  fdata::Matrix{Float64}
  vparadata::Vector{Float64}
  vperpdata::Vector{Float64}
  fmax::Float64
  pitchofpeak::Float64
  energyofpeakkev::Float64
end

function NBIDataVparaVperp(fname, massnumber=2; cutoffbelowvpara=-Inf, cutoffwidthvpara=1e100)

  fnb = transpose(h5read(fname, "C"))
  vparadata = h5read(fname, "VPAR")[1, :][:]
  vperpdata = h5read(fname, "VPERP")[:, 1][:]

  @assert size(fnb, 1) == length(vparadata)
  @assert size(fnb, 2) == length(vperpdata)

  cutoffmask = (0.5 .+ 0.5 .* erf.((vparadata .- cutoffbelowvpara) ./ cutoffwidthvpara))
  @assert minimum(cutoffmask) >= 0
  @assert maximum(cutoffmask) <= 1
  fnb .*= cutoffmask

  fmax, ind = findmax(fnb)
  fnb ./= fmax

  speedofpeak = sqrt(vparadata[ind[2]]^2 + vperpdata[ind[1]]^2)
  pitchofpeak = vparadata[ind[2]] / speedofpeak
  energyofpeakkeV = 0.5 * 1836mₑ * massnumber * speedofpeak^2 / 1000q₀

  return NBIDataVparaVperp(massnumber, fname, fnb, vparadata, vperpdata, fmax, pitchofpeak, energyofpeakkeV)
end

pitchofpeak(n::NBIDataVparaVperp) = n.pitchofpeak
energyofpeakkev(n::NBIDataVparaVperp) = n.energyofpeakkev
maxspeed(n::NBIDataVparaVperp) = sqrt(maximum(n.vparadata.^2 .+ n.vperpdata'.^2))

speedmin(n::NBIDataEnergyPitch) = n.speedmin
speedrange(n::NBIDataEnergyPitch) = n.speedrange
speedmin(n::NBIDataVparaVperp) = sqrt(minimum(n.vparadata.^2 .+ n.vperpdata'.^2))
speedrange(n::NBIDataVparaVperp) = diff(collect(extrema(sqrt.(n.vparadata.^2 .+ n.vperpdata'.^2))))[1]

pitchmin(n::NBIDataVparaVperp) = minimum(n.vparadata ./ sqrt.(n.vparadata.^2 .+ n.vperpdata'.^2))
pitchmax(n::NBIDataVparaVperp) = maximum(n.vparadata ./ sqrt.(n.vparadata.^2 .+ n.vperpdata'.^2))
pitchrange(n::NBIDataVparaVperp) = pitchmax(n) - pitchmin(n)

Base.size(n::AbstractNBIData) = size(n.fdata)

struct NBIInterpVelocitySpace{T} <: Function
  interp::T
  nbidata::NBIDataEnergyPitch
  keVperv²::Float64
  minpitch::Float64
  maxpitch::Float64
  minenergy::Float64
  maxenergy::Float64
end
speedofpeak(n::NBIInterpVelocitySpace) = sqrt(energyofpeakkev(n.nbidata) / n.keVperv²)
pitchofpeak(n::NBIInterpVelocitySpace) = pitchofpeak(n.nbidata)
maxspeed(n::NBIInterpVelocitySpace) = sqrt(maxenergy(n.nbidata) / n.keVperv²)

function prepend!(v::Vector, x::Number)
  resize!(v, length(v) + 1)
  v[2:end] .= v[1:end-1]
  v[1] = x
end

function NBIInterpVelocitySpace(nbidata::NBIDataEnergyPitch)
  mass = nbidata.massnumber * 1836mₑ
  v²perkeV = 1000q₀ * 2 / mass
  keVperv² = 1 / v²perkeV

  minpitch = minimum(nbidata.pdata)
  maxpitch = maximum(nbidata.pdata)
  minenergy = minimum(nbidata.edata)
  maxenergy = maximum(nbidata.edata)
  fdata_v = similar(nbidata.fdata)

  @turbo for (j, ekev) in enumerate(nbidata.edata)
    v = sqrt(ekev * v²perkeV)
    for i in eachindex(nbidata.pdata)
      fdata_v[i, j] = nbidata.fdata[i, j] / v # divide by v not mistake
    end
  end

  # here is the place to pad with zeros to get lienar interpolation at edges
  interp = linear_interpolation((nbidata.pdata, nbidata.edata), fdata_v)

  return NBIInterpVelocitySpace(interp, nbidata,
    keVperv², minpitch, maxpitch, minenergy, maxenergy)
end
function (nbi::NBIInterpVelocitySpace)(vz::T, v⊥::U) where {T, U}
  v² = vz^2 + v⊥^2
  pitch = vz / sqrt(v²)
  ekev = v² * nbi.keVperv²
  ekev > nbi.maxenergy && return zero(promote_type(T, U))
  ekev < nbi.minenergy && return zero(promote_type(T, U))
  pitch > nbi.maxpitch && return zero(promote_type(T, U))
  pitch < nbi.minpitch && return zero(promote_type(T, U))
  return nbi.interp(pitch, ekev)
end
(nbi::NBIInterpVelocitySpace)(vz⊥) = nbi.interp_vz⊥(vz⊥[1], vz⊥[2])


"""
    cacheic_fname(tuple)

Consistently generate a hash from the optimisation arguments to save and load files from.

...
# Arguments
- `tuple`: a tuple of arguments that uniquely (hopefully) determine the optimisation procedure
...

# Example
```julia
```
"""
function cacheic_fname(tuple)
  return "BlackBoxOptim_cache_$(hash(tuple)).jlso"
end

"""
    optctrl_generator(objective::F,nbidata::AbstractNBIData,nringbeams::Int,bboptimizemethod,targetfitness,traceinterval)whereF

This creates an optimisation control struct for internal use by BlackBoxOptim

...
# Arguments
- `objective::F`:  the objective function
- `nbidata::AbstractNBIData`: the data to fit turbo
- `nringbeams::Int`:  the number of ring-beams
- `bboptimizemethod`: the method used internally by BlackBoxOptim
- `targetfitness`: the target fitness for the fitting to achieve
- `traceinterval`: the interval that BlackBoxOptim prints info to screen (yes, this affects the optimisation procedure!!)
...

# Example
```julia
```
"""
function optctrl_generator(objective::F, nbidata::AbstractNBIData, nringbeams::Int,
    bboptimizemethod, targetfitness, traceinterval) where F
  fname = cacheic_fname((nbidata, nringbeams, bboptimizemethod, targetfitness, traceinterval))
  optctrl = if isfile(fname)
    @info "Using $(fname) for optimisation control"
    loaded = JLSO.load(fname)
    loaded[:optctrl]
  else
    @info "Cachfile $fname not found, so generating a new optimisation control"
    bbsetup(objective;
      SearchRange = (0.0, 1.0),
      NumDimensions = ncomponents_per_ringbeam * nringbeams,
      Method = bboptimizemethod,
      TargetFitness = targetfitness,
      TraceInterval = traceinterval)
  end
  return optctrl
end

"""
    differentialevolutionfitspecies(nbidata::AbstractNBIData,Π,Ω,numberdensity;nringbeams=100,bboptimizemethod=:adaptive_de_rand_1_bin,timelimithours=12,targetfitness=0.01,traceinterval=600.0,performoptimisation=true)::Vector{TSpecies}

Return a vector of RingBeam Species that best fit the data given the parameters

...
# Arguments
- `nbidata::AbstractNBIData`:
- `Π`: the plasma frequency of the whole energetic ion population
- `Ω`: the cyclotron frequency
- `numberdensity`: total number density of energetic ion population
Optional args:
- `nringbeams=100`: number of ring beams
- `bboptimizemethod=:adaptive_de_rand_1_bin`: used internally by BlackBoxOptim
- `timelimithours=12`:
- `targetfitness=0.01`:
- `traceinterval=600.0`: in seconds
- `performoptimisation=true::Vector{TSpecies}`: when false, this reads data from file and returns it
- `initialguessfilepath=nothing`: If not nothing then it must be a string to a file,
  e.g. "BlackBoxOptim_results_<hash>.jlso", which contains a previous optimisation result
  that will be used as the initial guess for this optimisation
...

# Example
```julia
```
"""
function differentialevolutionfitspecies(nbidata::AbstractNBIData, Π, Ω, numberdensity;
    nringbeams=100, bboptimizemethod=:adaptive_de_rand_1_bin,
    timelimithours=12, targetfitness=0.01, traceinterval=600.0,
    performoptimisation=true, initialguessfilepath=nothing)::Vector{TSpecies}

  cachehash = foldr(hash, (Π, Ω, numberdensity, nringbeams, bboptimizemethod);
                    init=hash(nbidata))

  mass = nbidata.massnumber * 1836mₑ
  v²perkeV = 1000q₀ * 2 / mass

  @info "Generating differential evolution species."

  smax = maxspeed(nbidata)
  maxvth = smax / 4

  pmin = pitchmin(nbidata)
  plen = pitchrange(nbidata)

  smin = speedmin(nbidata)
  srange = speedrange(nbidata)

  function speciesparams(x)
    @assert length(x) == ncomponents_per_ringbeam
    amp = sqrt(abs(x[1]))
    p = pmin + plen * x[2] # particle pitch [pmin, pmax]
    v = smin + srange * x[3] # particle speed
    uz = v * p
    u⊥ = v * sqrt(1 - p^2)
    vthz = maxvth * x[4]
    vth⊥ = maxvth * x[5]
    return (amp, vthz, uz, vth⊥, u⊥, p, v)
  end

  function speciesscalar(k, x; Π0=Π)
    @assert k > 0
    l = ncomponents_per_ringbeam * (k-1)
    (amp, vthz, uz, vth⊥, u⊥, _, _) = speciesparams(x[l+1:l+ncomponents_per_ringbeam])
    return SeparableVelocitySpecies(Π0 * amp, Ω, FBeam(vthz, uz), FRing(vth⊥, u⊥))
  end
  function speciesvector(x; Π0=Π)
    dfs = Vector{TSpecies}(undef, nringbeams)
    @views @inbounds for k in 1:nringbeams
      dfs[k] = speciesscalar(k, x; Π0=Π0)
    end
    return dfs
  end

  M = zeros(Float64, (size(nbidata)..., Threads.nthreads()))
  M0 = zeros(Float64, size(nbidata))

  function reducematrices!(M0, M)
    @inbounds @turbo for i in eachindex(M0)
      M0[i] = 0
    end
    @inbounds @turbo for k in 1:size(M, 3), j in 1:size(M, 2), i in 1:size(M, 1)
      M0[i, j] += M[i, j, k]
      M[i, j, k] = 0
    end
    return M0
  end

  function objectivefit(x, nbidata::NBIDataEnergyPitch)
    @threads for k in 1:nringbeams
      s = speciesscalar(k, x)
      ρ = density(s, mass)
      Mlocal = @view M[:, :, Threads.threadid()]
      @turbo for (j, ekev) in enumerate(nbidata.edata)
        v = sqrt(ekev * v²perkeV)
        for i in eachindex(nbidata.pdata, nbidata.sqrt1_p²data)
          pitch = nbidata.pdata[i]
          sqrt1_p² = nbidata.sqrt1_p²data[i]
          vz = v * pitch
          v⊥ = v * sqrt1_p² # sqrt(1 - pitch^2)
          @inbounds Mlocal[i, j] += s(vz, v⊥) * ρ
        end
      end
    end
    M0 = reducematrices!(M0, M)
    return M0
  end
  function objectivefit(x, nbidata::NBIDataVparaVperp)
    @threads for k in 1:nringbeams
      s = speciesscalar(k, x)
      ρ = density(s, mass)
      Mlocal = @view M[:, :, Threads.threadid()]
      @turbo for (j, vz) in enumerate(nbidata.vparadata)
        for (i, v⊥) in enumerate(nbidata.vperpdata)
          @inbounds Mlocal[i, j] += s(vz, v⊥) * ρ
        end
      end
    end
    M0 = reducematrices!(M0, M)
    return M0
  end

  function rescalebyjacobians!(A, nbidata::NBIDataEnergyPitch)
    @turbo for (j, ekev) in enumerate(nbidata.edata)
      v = sqrt(ekev * v²perkeV)
      #=
      NUBEAM data are in (pitch, energy) and I want (v_parallel, v_perp)
      and f_NUBEAM(pitch, energy) / v \propto f_LMV(v_parallel, v_perp)
      NUBEAM data are in units of v^-2, LMV in v^-3.
      =#
      for i in eachindex(nbidata.pdata)
        @inbounds A[i, j] *= v
      end
    end
    return A
  end
  rescalebyjacobians!(A, nbidata::NBIDataVparaVperp) = A

  fnorm = norm(nbidata.fdata)

  function objective(x)
    A = objectivefit(x, nbidata)
    rescalebyjacobians!(A, nbidata)
    maxA = maximum(A)
    @turbo for i in eachindex(A, nbidata.fdata)
      A[i] /= maxA
      A[i] -= nbidata.fdata[i]
    end
    return norm(A) / fnorm
  end

  optctrl = optctrl_generator(objective, nbidata, nringbeams, bboptimizemethod, targetfitness,
                             traceinterval)

  ix = rand(ncomponents_per_ringbeam * nringbeams)

  t = @elapsed res = if performoptimisation
    initialguess = if !isnothing(initialguessfilepath)
      @info "Starting optimisation state from $initialguessfilepath"
      loaded = JLSO.load(initialguessfilepath)
      initialguessres = loaded[:results]
      best_candidate(initialguessres)
    end
    bboptimize(optctrl, initialguess;
      TraceMode = :compact,
      MaxTime = 3600 * timelimithours,
      MaxSteps = 1_000_000,
      MaxFuncEvals = timelimithours == 0 ? 0 : Int(maxintfloat()))
  else
    loaded = JLSO.load("BlackBoxOptim_results_$(cachehash).jlso")
    loaded[:results]
  end

  xbest = best_candidate(res)
  fitness = best_fitness(res)
  @info "Fitness = $fitness"

  performoptimisation && @info "Fitness achieved is $fitness in $(t / 3600) hours."
  dfs = speciesvector(best_candidate(res))

  nprenorm = sum(density(i, mass) for i in dfs)
  nbi_species = speciesvector(xbest, Π0=Π * sqrt(numberdensity / nprenorm))
  nnbi = sum(density(i, mass) for i in nbi_species)
  @assert nnbi ≈ numberdensity

  JLSO.save("BlackBoxOptim_results_$(cachehash).jlso",
    :optctrl => optctrl, :results => res, :xbest => xbest, :nbi_species => nbi_species, :fitness => fitness)

  JLSO.save("BlackBoxOptim_nbi_species_$(cachehash).jlso",
    :nbi_species => nbi_species,
    :maxspeed => smax)

  if typeof(nbidata) <: NBIDataEnergyPitch
  JLSO.save("BlackBoxOptim_plotdata_$(cachehash).jlso",
    :nbi_species => nbi_species,
    :maxspeed => smax,
    :edata => nbidata.edata,
    :pdata => nbidata.pdata,
    :fdata => nbidata.fdata,
    :scaledfit => deepcopy(objectivefit(xbest, nbidata)),
    :fit => rescalebyjacobians!(deepcopy(objectivefit(xbest, nbidata)), nbidata))
  elseif typeof(nbidata) <: NBIDataVparaVperp
  JLSO.save("BlackBoxOptim_plotdata_$(cachehash).jlso",
    :nbi_species => nbi_species,
    :maxspeed => smax,
    :vparadata => nbidata.vparadata,
    :vperpdata => nbidata.vperpdata,
    :fdata => nbidata.fdata,
    :scaledfit => deepcopy(objectivefit(xbest, nbidata)),
    :fit => rescalebyjacobians!(deepcopy(objectivefit(xbest, nbidata)), nbidata))
  end

  JLSO.save(cacheic_fname((nbidata, nringbeams, bboptimizemethod, targetfitness, traceinterval)),
            :optctrl => optctrl, :xbest => xbest)

  return nbi_species
end

density(Π::Real, mass) = Π^2 / q₀^2 * mass * LinearMaxwellVlasov.ϵ₀
density(s, mass) = density(s.Π, mass)

function couplednbispecies(nbiinterp::NBIInterpVelocitySpace, Π, Ω)
  vinj = speedofpeak(nbiinterp)
  pitch = pitchofpeak(nbiinterp)

  vzinj = vinj * pitch
  v⊥inj = vinj * sqrt(1 - pitch^2)
  smax = maxspeed(nbiinterp)
  fvz⊥ = FCoupledVelocityNumerical(nbiinterp, (vzinj, v⊥inj), 0.0, smax, autonormalise=true)

  species = CoupledVelocitySpecies(Π, Ω, fvz⊥)
  return species
end

end
