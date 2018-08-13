module StaticOptim
using Parameters, ForwardDiff, StaticArrays
using Statistics: middle
using LinearAlgebra, Printf
import NaNMath
import Base.show
export soptimize, sroot

include("soptimize.jl")
end
