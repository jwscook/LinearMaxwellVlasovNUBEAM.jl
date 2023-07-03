using Plots, JLSO, LinearMaxwellVlasov, StatsPlots
include("NBI.jl")
using .NBI

fname = String(ARGS[1]);

loaded = JLSO.load(fname)
nbi_species = loaded[:nbi_species];

x = loaded[:edata];
y = loaded[:pdata];

z0 = loaded[:fdata];

z1 = loaded[:fit];
z1 ./= maximum(z1);

z2 = loaded[:scaledfit];
z2 ./= maximum(z2);

h0 = heatmap(x, y, z0, title="data")
h1 = heatmap(x, y, z1, title="fit")
h2 = heatmap(x, y, z1 .- z0, title="fit - data")
h3 = heatmap(x, y, z2, title="pure f")
h = plot(h0, h1, h2, h3, layout = @layout [a b; c d])
savefig(h, fname * ".data.pdf")


smax = loaded[:maxspeed];
N = 1024
vzs = collect(range(-smax, stop=smax, length=N))
v⊥s = collect(range(0, stop=smax, length=N))

z3 = zeros(N, N)
for species in loaded[:nbi_species]
  ρ = NBI.density(species, 1836 * 2 * LinearMaxwellVlasov.mₑ)
  for (j, v⊥) in enumerate(v⊥s), (i, vz) in enumerate(vzs)
    z3[i, j] += species(vz, v⊥) * ρ
  end
end

h = heatmap(v⊥s, vzs, z3)
xlabel!(h, "v⊥ [m/s]")
ylabel!(h, "vz [m/s]")
savefig(h, fname * ".nbi_species.pdf")

