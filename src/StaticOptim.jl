module StaticOptim
using Parameters, ForwardDiff, StaticArrays
using Statistics: middle
using LinearAlgebra, Printf
import NaNMath
import Base.show
export soptimize, sroot, constrained_soptimize

include("soptimize.jl")
end
