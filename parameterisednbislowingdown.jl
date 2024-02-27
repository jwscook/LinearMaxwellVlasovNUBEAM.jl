using Plots, SpecialFunctions
v0 = 1.0
p0 = 0.45
vthcutoff = 0.01
vth = 0.1
vthz0 = 0.1
vth⊥0 = 0.1
vz0 = p0 * v0
v⊥0 = v0 * sqrt(1 - p0^2)

vzs = -2v0:0.01:2v0
v⊥s = 0:0.01:2v0

v = @. sqrt(v⊥s'^2 + vzs^2)
p = @. vzs / v

# for NBI slowing down
vthz = @. 0.05 / (vthz0 + v⊥s')
fv⊥ = @. v^2 / (v^3 + v0^3)
fvz = @. exp(-(vzs-vz0)^2/vthz^2)
fcutoff = @. (0.5 + 0.5*erf((v0 - v)/vthcutoff))
f = @. fvz * fv⊥ * fcutoff
heatmap(v⊥s, vzs, f)

# for just one arc of a fresh NBI beam
vthz0 = 0.2
vth⊥0 = 0.2
fv = @. (v/v0)^6 * exp(-(vzs-vz0)^2/vthz0^2 - (v⊥s' - v⊥0)^2 / vth⊥0^2)
fcutoff = @. (0.5 + 0.5*erf((v0 - v)/vthcutoff))
f = @. fv * fcutoff
heatmap(v⊥s, vzs, f)


fv = @. exp(v/v0/param[1]) * # slope up
  exp(- (p - param[2])^2 / param[3]^2) * # pitch shape
  (0.5 + 0.5*erf((v0 * param[4] - v)/v0/param[5])) # slope back down


