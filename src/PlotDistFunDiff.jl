using Plots, JLSO

fname = String(ARGS[1]);

loaded = JLSO.load(fname)

x = try
  x1 = loaded[:edata];
  @info "Plotting energy and pitch"
  x1
catch err
  x1 = loaded[:vperpdata];
  @info "Plotting vpara and vperp"
  x1
end
y = try
  loaded[:pdata];
catch err
  loaded[:vparadata];
end

z0 = loaded[:fdata];

z1 = loaded[:fit];
z1 ./= maximum(z1);

z2 = loaded[:scaledfit];
z2 ./= maximum(z2);

h0 = heatmap(x, y, log10.(z0), title="data")
h1 = heatmap(x, y, log10.(z1), title="fit")
h2 = heatmap(x, y, z1 .- z0, title="fit - data")
h3 = heatmap(x, y, log10.(z2), title="pure f")
h = plot(h0, h1, h2, h3, layout = @layout [a b; c d])
savefig(h, fname * ".pdf")


