# StaticOptim

[![Build Status](https://travis-ci.org/aaowens/StaticOptim.jl.svg?branch=master)](https://travis-ci.org/aaowens/StaticOptim.jl)

[![Coverage Status](https://coveralls.io/repos/aaowens/StaticOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/aaowens/StaticOptim.jl?branch=master)

[![codecov.io](http://codecov.io/github/aaowens/StaticOptim.jl/coverage.svg?branch=master)](http://codecov.io/github/aaowens/StaticOptim.jl?branch=master)

# Example:
```
julia> using StaticArrays, BenchmarkTools, StaticOptim

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

julia> @btime soptimize(rosenbrock, $sx)
  5.928 μs (0 allocations: 0 bytes)
Results of Static Optimization Algorithm
 * Minimizer: [1.00000000000079,1.0000000000014035]
 * Minimum: [3.7402786691745805e-24]
 * |Df(x)|: [8.03609499330041e-11]
 * Hf(x): [809.094042962608,-403.46938387131536,-403.4693838713215,201.6967228908349]
 * Number of iterations: [58]
 * Converged: [true]
```
# Example of univariate derivative based optimization with numbers, not StaticArrays
```
julia> using StaticOptim, BenchmarkTools

julia> function U(x)
           x = max(0., x)
           log(x)
       end
U (generic function with 1 method)

julia> f(h) = -(U(0.2 + h) + U(1 - h))
f (generic function with 1 method)

julia> @btime soptimize(f, 0.9)
  886.531 ns (0 allocations: 0 bytes)
Results of Static Optimization Algorithm
 * Minimizer: [0.39999999998562896]
 * Minimum: [1.0216512475319814]
 * |Df(x)|: [7.983924632526396e-11]
 * Hf(x): [5.555569904413021]
 * Number of iterations: [6]
 * Converged: [true]
```
