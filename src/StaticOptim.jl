module StaticOptim
using Parameters, ForwardDiff, StaticArrays
import NaNMath
import Base.show
export soptimize

include("soptimize.jl")
end
