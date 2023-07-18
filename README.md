# LinearMaxwellVlasovNUBEAM.jl

This project contains code to fit ring-beam distribution functions to NUBEAM data
and then to calculate growth rates of these ring-beams with respect to any unstable
hot plasma waves, in particular the fast Alfven wave via the magnetoacoustic
cyclotron instability (MCI), which is responsible for ion cyclotron emission (ICE).

The code is driven by `ICE.jl` i.e. `julia --proj src/ICE.jl` where arguments can be
given to control certain parameters (see the top of the file itself). The code does
the following:
 - fits a number of ring-beam sub-populations of energetic ions to NUBEAM data,
 - calculates the growth rates of any unstable modes,
 - plots and saves the results.

