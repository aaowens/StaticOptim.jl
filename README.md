# StaticOptim

[![Build Status](https://travis-ci.org/aaowens/StaticOptim.jl.svg?branch=master)](https://travis-ci.org/aaowens/StaticOptim.jl)

[![Coverage Status](https://coveralls.io/repos/aaowens/StaticOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/aaowens/StaticOptim.jl?branch=master)

[![codecov.io](http://codecov.io/github/aaowens/StaticOptim.jl/coverage.svg?branch=master)](http://codecov.io/github/aaowens/StaticOptim.jl?branch=master)

# Example:
```
julia> using StaticOptim

julia> using StaticArrays

julia> sx = @SVector ones(2)
2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:
 1.0
 1.0

julia> sx = 3.2 * sx
2-element StaticArrays.SArray{Tuple{2},Float64,1,2}:
 3.2
 3.2

julia> rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rosenbrock (generic function with 1 method)

julia> res = soptimize(rosenbrock, sx)
Results of Static Optimization Algorithm
 * Minimizer: [1.0000000000007898,1.0000000000014029]
 * Minimum: [3.7477721082170814e-24]
 * |Df(x)|: [8.045984853069932e-11]
 * Hf(x): [809.0962085236879,-403.47047521120845,-403.47047521102013,201.69727239400734]
 * Number of iterations: [58]
```
