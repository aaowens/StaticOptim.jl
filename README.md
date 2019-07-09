# StaticOptim

[![Build Status](https://travis-ci.org/aaowens/StaticOptim.jl.svg?branch=master)](https://travis-ci.org/aaowens/StaticOptim.jl)

[![Coverage Status](https://coveralls.io/repos/aaowens/StaticOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/aaowens/StaticOptim.jl?branch=master)

[![codecov.io](http://codecov.io/github/aaowens/StaticOptim.jl/coverage.svg?branch=master)](http://codecov.io/github/aaowens/StaticOptim.jl?branch=master)

This package implements scalar and multivariate optimization routines optimized for low
dimensional problems and cheap function calls. It also has two univariate root-finding
routines: a modified Newton method and a bisection method. All functions except bisection
use ForwardDiff to compute derivatives. They should not allocate much if the input function does not,
thanks in part to the stack allocated gradient and hessian methods for StaticArrays in ForwardDiff.

The optimization uses the Newton method with a quadratic or cubic backtracking linesearch
inspired by LineSearches.jl. Root-finding with an initial guess is done using a modified
Newton method which may not be very robust, but is fast in the problems I've tried.
Root-finding with a bracket as a 2-tuple is done by bisection.

# Example:
```
julia> using StaticArrays, BenchmarkTools, StaticOptim

julia> sx = @SVector ones(2)
2-element SArray{Tuple{2},Float64,1,2}:
 1.0
 1.0

julia> sx = 3.2 * sx
2-element SArray{Tuple{2},Float64,1,2}:
 3.2
 3.2

julia> rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rosenbrock (generic function with 1 method)

julia> @btime soptimize(rosenbrock, $sx)
  2.184 μs (27 allocations: 2.91 KiB)
Results of Static Optimization Algorithm
 * Initial guess: [3.2,3.2]
 * Minimizer: [1.0000000000000009,1.0000000000000009]
 * Minimum: [7.967495142732219e-29]
 * ∇f(x): [3.5704772471945065e-13,-1.7763568394002505e-13]
 * Hf(x) is size [2,2]
 * Number of iterations: [25]
 * Number of function calls: [56]
 * Number of gradient calls: [25]
 * Converged: [true]


 julia> @btime soptimize(rosenbrock, $sx, bto = StaticOptim.Order3())
   2.189 μs (27 allocations: 2.91 KiB)
 Results of Static Optimization Algorithm
  * Initial guess: [3.2,3.2]
  * Minimizer: [1.0000000000000009,1.0000000000000009]
  * Minimum: [7.967495142732219e-29]
  * ∇f(x): [3.5704772471945065e-13,-1.7763568394002505e-13]
  * Hf(x) is size [2,2]
  * Number of iterations: [25]
  * Number of function calls: [56]
  * Number of gradient calls: [25]
  * Converged: [true]


 julia> rosenbrock2(x) =  (1.05 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
 rosenbrock2 (generic function with 1 method)

 julia> res = soptimize(rosenbrock, sx)
 Results of Static Optimization Algorithm
  * Initial guess: [3.2,3.2]
  * Minimizer: [1.0000000000000009,1.0000000000000009]
  * Minimum: [7.967495142732219e-29]
  * ∇f(x): [3.5704772471945065e-13,-1.7763568394002505e-13]
  * Hf(x) is size [2,2]
  * Number of iterations: [25]
  * Number of function calls: [56]
  * Number of gradient calls: [25]
  * Converged: [true]


 julia> @btime soptimize(rosenbrock2, $sx)
   1.929 μs (25 allocations: 2.69 KiB)
 Results of Static Optimization Algorithm
  * Initial guess: [3.2,3.2]
  * Minimizer: [1.0500000000744594,1.102500000154429]
  * Minimum: [5.918917770173918e-21]
  * ∇f(x): [9.619482988520233e-10,-3.871569731472846e-10]
  * Hf(x) is size [2,2]
  * Number of iterations: [23]
  * Number of function calls: [49]
  * Number of gradient calls: [23]
  * Converged: [true]


 julia> @btime soptimize(rosenbrock2, $res.minimizer)
   301.447 ns (5 allocations: 512 bytes)
 Results of Static Optimization Algorithm
  * Initial guess: [1.0000000000000009,1.0000000000000009]
  * Minimizer: [1.0499999999999965,1.1024999999999925]
  * Minimum: [1.7552155141167513e-29]
  * ∇f(x): [8.615330671091183e-14,-4.440892098500626e-14]
  * Hf(x) is size [2,2]
  * Number of iterations: [3]
  * Number of function calls: [5]
  * Number of gradient calls: [3]
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

julia> @btime soptimize(f, 0.5)
  350.079 ns (1 allocation: 96 bytes)
Results of Static Optimization Algorithm
 * Initial guess: [0.5]
 * Minimizer: [0.4]
 * Minimum: [1.0216512475319812]
 * ∇f(x): [2.220446049250313e-16]
 * Hf(x) is size []
 * Number of iterations: [4]
 * Number of function calls: [7]
 * Number of gradient calls: [4]
 * Converged: [true]


julia> using Optim

julia> @btime optimize(f, 0.1, 0.9) # Maybe a better idea
  425.819 ns (2 allocations: 176 bytes)
Results of Optimization Algorithm
 * Algorithm: Brent's Method
 * Search Interval: [0.100000, 0.900000]
 * Minimizer: 4.000000e-01
 * Minimum: 1.021651e+00
 * Iterations: 8
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16): true
 * Objective Function Calls: 9

julia> @btime soptimize(f, 0.45) # This converges very fast when the guess is close
  355.538 ns (1 allocation: 96 bytes)
Results of Static Optimization Algorithm
 * Initial guess: [0.45]
 * Minimizer: [0.39999999999999997]
 * Minimum: [1.0216512475319812]
 * ∇f(x): [-2.220446049250313e-16]
 * Hf(x) is size []
 * Number of iterations: [4]
 * Number of function calls: [7]
 * Number of gradient calls: [4]
 * Converged: [true]
```

# Example of 1st order root-finding  
```
julia> up(c) = c <= 0 ? Inf*c : 1/c
up (generic function with 1 method)

julia> f(a) = up(2 - a) - .96up(2 + a)
f (generic function with 1 method)

julia> @btime sroot(f, 0.5)
  179.253 ns (1 allocation: 48 bytes)
(x = -0.04081632661278052, fx = -4.027922440030807e-11, isroot = true, iter = 3)

julia> @btime sroot(f, (-0.5, 0.5))
  252.507 ns (2 allocations: 64 bytes)
(x = -0.040816307067871094, fx = 9.54071677217172e-9, isroot = true, iter = 20, ismaxiter = false)
```

# Example of non-linear equation solving
```
julia> using StaticArrays, BenchmarkTools, StaticOptim

julia> const w = 3.
3.0

julia> const beta = 0.96
0.96

julia> const R = 1.01
1.01

julia> const alpha_h = 1.5
1.5

julia> uc(c) = 1/c
uc (generic function with 1 method)

julia> uh(h) = alpha_h/(1 - h)
uh (generic function with 1 method)

julia> function eulerfun(x)
           a, h1, h2 = x[1], x[2], x[3]
           c1 = w*h1 - a
           c2 = w*h2 + R*a
           (h1 >= 1 || h2 >= 1) && return Inf*x
           c1 <= 0 && return Inf*x
           out1 = uc(c1) - beta*R*uc(c2)
           out2 = w*uc(c1) - uh(h1)
           out3 = w*uc(c2) - uh(h2)
           SVector(out1, out2, out3)
       end
eulerfun (generic function with 1 method)

julia> x = SVector(0., 0.5, 0.5)
3-element SArray{Tuple{3},Float64,1,3}:
 0.0
 0.5
 0.5

julia> @btime sroot(eulerfun, $x)
  2.385 μs (12 allocations: 1.45 KiB)
Results of Static Optimization Algorithm
 * Initial guess: [0.0,0.5,0.5]
 * Minimizer: [-0.046069913119393756,0.3907860173760137,0.4093061224496989]
 * Minimum: [3.551982210586524e-23]
 * ∇f(x): [-2.5470608653058333e-11,-6.787125184864095e-12,-1.1236500463940956e-10]
 * Hf(x) is size [3,3]
 * Number of iterations: [6]
 * Number of function calls: [11]
 * Number of gradient calls: [6]
 * Converged: [true]
```

 # Example of constrained optimization
```
julia> function U(h)
          h1, h2 = h[1], h[2]
          h1 >= 1 && return -Inf*one(h1)
          h2 >= 1 && return -Inf*one(h1)
          c = 2h1 + 1.5h2 + 0.1
          c <= 0 && return -Inf*one(h1)
          log(c) + log(1 - h1) + log(1 - h2)
             end
U (generic function with 1 method)

julia> l = SVector{2}(0., 0.4)
2-element SArray{Tuple{2},Float64,1,2}:
 0.0
 0.4

julia> sx = SVector{2}(0.7, 0.7)
2-element SArray{Tuple{2},Float64,1,2}:
 0.7
 0.7

julia> res = constrained_soptimize(x -> -U(x), sx, lower = l)
Results of Static Optimization Algorithm
 * Initial guess: [0.7,0.7]
 * Minimizer: [0.32500000012427305,0.4]
 * Minimum: [0.6037636194252598]
 * ∇f(x): [5.455058627035214e-10,0.5555555557601204]
 * Hf(x) is size [2,2]
 * Number of iterations: [5]
 * Number of function calls: [9]
 * Number of gradient calls: [5]
 * Converged: [true]


```
