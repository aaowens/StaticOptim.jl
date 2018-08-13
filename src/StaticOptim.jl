module StaticOptim
using Parameters, ForwardDiff, StaticArrays
using Statistics: middle
import NaNMath
import Base.show
export soptimize, snewton, bisection

include("soptimize.jl")
end
