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


@auto_hash_equals struct NBIData
  massnumber::Int
  fname::String
  fdata::Matrix{Float64}
  pdata::Vector{Float64}
  sqrt1_p²data::Vector{Float64}
  edata::Vector{Float64}
  fmax::Float64
end

function NBIData(fname, massnumber=2;
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

  return NBIData(massnumber, fname, fnb, pitch, sqrt.(1 .- pitch.^2), energy, fmax)
end

peakpitch(n::NBIData) = n.pdata[findmax(n.fdata)[2][1]]
peakenergy(n::NBIData) = n.edata[findmax(n.fdata)[2][2]]
maxenergy(n::NBIData) = n.edata[end]
maxspeed(n::NBIData) = sqrt(maxenergy(n) * 1000q₀ * 2 / (n.massnumber * 1836mₑ))

Base.size(n::NBIData) = size(n.fdata)

struct NBIInterpVelocitySpace{T} <: Function
  interp::T
  nbidata::NBIData
  keVperv²::Float64
  minpitch::Float64
  maxpitch::Float64
  minenergy::Float64
  maxenergy::Float64
end
peakspeed(n::NBIInterpVelocitySpace) = sqrt(peakenergy(n.nbidata) / n.keVperv²)
peakpitch(n::NBIInterpVelocitySpace) = peakpitch(n.nbidata)
maxspeed(n::NBIInterpVelocitySpace) = sqrt(maxenergy(n.nbidata) / n.keVperv²)

function prepend!(v::Vector, x::Number)
  resize!(v, length(v) + 1)
  v[2:end] .= v[1:end-1]
  v[1] = x
end

function NBIInterpVelocitySpace(nbidata::NBIData)
  mass = nbidata.massnumber * 1836 * mₑ
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


cachehash_ic(nbidata, nringbeams) = hash(nringbeams, hash(nbidata))
function cacheic_fname(nbidata, nringbeams)
  return "BlackBoxOptim_xbest_$(cachehash_ic(nbidata, nringbeams)).jlso"
end
function default_initialconditions_generator(nbidata::NBIData, nringbeams::Int)
  fname = cacheic_fname(nbidata, nringbeams)
  ic = if isfile(fname)
    @info "Using $(fname) for initial condition"
    loaded = JLSO.load(fname)
    loaded[:xbest]
  else
    @info "Cachfile $fname not found"
    rand(nringbeams * ncomponents_per_ringbeam)
  end
  return ic
end

function differentialevolutionfitspecies(nbidata::NBIData, Π, Ω, numberdensity,
    initialconditions_generator::F=default_initialconditions_generator;
    nringbeams=100, bboptimizemethod=:adaptive_de_rand_1_bin,
    timelimithours=12, targetfitness=0.01, prenormop::T=identity,
    traceinterval=600.0)::Vector{TSpecies} where {F, T}

  cachehash = foldr(hash, (Π, Ω, numberdensity, nringbeams, bboptimizemethod,
     timelimithours, targetfitness); init=hash(nbidata))

  mass = nbidata.massnumber * 1836 * mₑ
  v²perkeV = 1000q₀ * 2 / mass

  @info "Generating differential evolution species."

  smax = maxspeed(nbidata)
  maxvth = smax / 4

  pmin = minimum(nbidata.pdata)
  plen = maximum(nbidata.pdata) - pmin

  vmin = sqrt(minimum(nbidata.edata) * v²perkeV)
  vlen = sqrt(maximum(nbidata.edata) * v²perkeV) - vmin

  function speciesparams(x)
    @assert length(x) == ncomponents_per_ringbeam
    amp = sqrt(abs(x[1]))
    p = pmin + plen * x[2] # particle pitch [pmin, pmax]
    v = vmin + vlen * x[3] # particle speed
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
  function objectivefit(x)
    @batch per=thread for k in 1:nringbeams
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
    @inbounds @turbo for i in eachindex(M0)
      M0[i] = 0
    end
    @inbounds @turbo for k in 1:size(M, 3), j in 1:size(M, 2), i in 1:size(M, 1)
      M0[i, j] += M[i, j, k]
      M[i, j, k] = 0
    end
    return M0
  end

  function rescalebyjacobians!(A)
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

  fnorm = norm(prenormop.(nbidata.fdata))

  function objective(x)
    A = objectivefit(x)
    rescalebyjacobians!(A)
    maxA = maximum(A)
    @turbo for i in eachindex(A, nbidata.fdata)
      A[i] /= maxA
      A[i] = prenormop(A[i])
      A[i] -= prenormop(nbidata.fdata[i])
    end
    return norm(A) / fnorm
  end

  ic = initialconditions_generator(nbidata, nringbeams)

  t = @elapsed res = bboptimize(objective, ic;
    SearchRange = (0.0, 1.0),
    NumDimensions = ncomponents_per_ringbeam * nringbeams,
    Method = bboptimizemethod,
    TraceInterval = traceinterval,
    TraceMode = :compact,
    TargetFitness = targetfitness,
    MaxTime = 3600 * timelimithours,
    MaxSteps = 1_000_000,
    MaxFuncEvals = timelimithours == 0 ? 0 : Int(maxintfloat()))
  xbest = best_candidate(res)
  fitness = best_fitness(res)

  @info "Fitness achieved is $fitness in $(t / 3600) hours."
  dfs = speciesvector(best_candidate(res))

  nprenorm = sum(density(i, mass) for i in dfs)
  nbi_species = speciesvector(xbest, Π0=Π * sqrt(numberdensity / nprenorm))
  nnbi = sum(density(i, mass) for i in nbi_species)
  @assert nnbi ≈ numberdensity

  JLSO.save("BlackBoxOptim_results_$(cachehash).jlso",
    :xbest => xbest, :nbi_species => nbi_species, :fitness => fitness)

  JLSO.save("BlackBoxOptim_nbi_species_$(cachehash).jlso",
    :nbi_species => nbi_species,
    :maxspeed => smax)

  JLSO.save("BlackBoxOptim_plotdata_$(cachehash).jlso",
    :nbi_species => nbi_species,
    :maxspeed => smax,
    :edata => nbidata.edata,
    :pdata => nbidata.pdata,
    :fdata => nbidata.fdata,
    :scaledfit => deepcopy(objectivefit(xbest)),
    :fit => rescalebyjacobians!(deepcopy(objectivefit(xbest))))

  JLSO.save(cacheic_fname(nbidata, nringbeams), :xbest => xbest)

  return nbi_species
end

density(Π::Real, mass) = Π^2 / q₀^2 * mass * LinearMaxwellVlasov.ϵ₀
density(s, mass) = density(s.Π, mass)

function couplednbispecies(nbiinterp::NBIInterpVelocitySpace, Π, Ω)
  vinj = peakspeed(nbiinterp)
  pitch = peakpitch(nbiinterp)

  vzinj = vinj * pitch
  v⊥inj = vinj * sqrt(1 - pitch^2)
  smax = maxspeed(nbiinterp)
  fvz⊥ = FCoupledVelocityNumerical(nbiinterp, (vzinj, v⊥inj), 0.0, smax, autonormalise=true)

  species = CoupledVelocitySpecies(Π, Ω, fvz⊥)
  return species
end

end
