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
