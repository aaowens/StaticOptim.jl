module StaticOptim
using Parameters, ForwardDiff, StaticArrays
using Statistics: middle
import NaNMath
import Base.show
export soptimize, sroot

include("soptimize.jl")
end
