#= ==========================================================================================
=============================================================================================
preamble
=============================================================================================
========================================================================================== =#

# Environment
using Pkg;
Pkg.activate("environment");

# numerical analysis
using SparseArrays 

# Graphics
using CairoMakie

# Data management
using Dates
(!isdir("anims")) ? (mkdir("anims")) : nothing; (!isdir("anims/$(today())")) ? (mkdir("anims/$(today())")) : nothing;
(!isdir("figs")) ? (mkdir("figs")) : nothing; (!isdir("figs/$(today())")) ? (mkdir("figs/$(today())")) : nothing;
