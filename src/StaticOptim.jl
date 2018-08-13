module StaticOptim
using Parameters, ForwardDiff, StaticArrays
import NaNMath
import Base.show
export soptimize, snewton

include("soptimize.jl")
end
