# StaticOptim

[![Build Status](https://travis-ci.org/aaowens/StaticOptim.jl.svg?branch=master)](https://travis-ci.org/aaowens/StaticOptim.jl)

[![Coverage Status](https://coveralls.io/repos/aaowens/StaticOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/aaowens/StaticOptim.jl?branch=master)

[![codecov.io](http://codecov.io/github/aaowens/StaticOptim.jl/coverage.svg?branch=master)](http://codecov.io/github/aaowens/StaticOptim.jl?branch=master)

This package implements scalar and multivariate optimization routines optimized for low
dimensional problems and cheap function calls. It also has two univariate root-finding
routines: a modified Newton method and a bisection method. All functions except bisection
use ForwardDiff to compute derivatives. They should not allocate if the input function does not,
thanks in part to the stack allocated gradient methods for StaticArrays in ForwardDiff.

The optimization uses the BFGS method with a quadratic or cubic backtracking linesearch
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
  1.934 μs (2 allocations: 160 bytes)
Results of Static Optimization Algorithm
 * Initial guess: [3.2,3.2]
 * Minimizer: [0.9999999999990606,0.9999999999980389]
 * Minimum: [1.5610191722141176e-24]
 * Hf(x): [801.6874976886638,-399.8345645795701,-399.83456457957504,199.9124176978296]
 * Number of iterations: [31]
 * Number of function calls: [43]
 * Number of gradient calls: [35]
 * Converged: [true]

# You can use the cubic linesearch, but it isn't as efficient here
julia> @btime soptimize(rosenbrock, $sx, bto = StaticOptim.Order3())
  4.288 μs (2 allocations: 160 bytes)
Results of Static Optimization Algorithm
 * Initial guess: [3.2,3.2]
 * Minimizer: [1.00000000000079,1.0000000000014035]
 * Minimum: [3.7402786691745805e-24]
 * Hf(x): [809.094042962608,-403.46938387131536,-403.4693838713215,201.6967228908349]
 * Number of iterations: [58]
 * Number of function calls: [101]
 * Number of gradient calls: [75]
 * Converged: [true]

 # Use the arguments to speed computation
 # For example, if we've already solved for the optimum of a similar function,
 # like when we are iterating on small parameter changes, we can pass the old guess
 # and the old hessian
 julia> rosenbrock2(x) =  (1.05 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
 rosenbrock2 (generic function with 1 method)

 julia> res = soptimize(rosenbrock, sx)
 Results of Static Optimization Algorithm
  * Initial guess: [3.2,3.2]
  * Minimizer: [0.9999999999990606,0.9999999999980389]
  * Minimum: [1.5610191722141176e-24]
  * Hf(x): [801.6874976886638,-399.8345645795701,-399.83456457957504,199.9124176978296]
  * Number of iterations: [31]
  * Number of function calls: [43]
  * Number of gradient calls: [35]
  * Converged: [true]


 julia> @btime soptimize(rosenbrock2, $sx)
   1.968 μs (2 allocations: 160 bytes)
 Results of Static Optimization Algorithm
  * Initial guess: [3.2,3.2]
  * Minimizer: [1.0500000000007772,1.1025000000017364]
  * Minimum: [1.6930927178305966e-24]
  * Hf(x): [883.1297900218369,-419.5702919718202,-419.57029197179963,199.78781023963182]
  * Number of iterations: [31]
  * Number of function calls: [45]
  * Number of gradient calls: [36]
  * Converged: [true]


 julia> @btime soptimize(rosenbrock2, $res.minimizer)
   588.221 ns (2 allocations: 160 bytes)
 Results of Static Optimization Algorithm
  * Initial guess: [0.9999999999990606,0.9999999999980389]
  * Minimizer: [1.0500000001310041,1.1025000002712262]
  * Minimum: [1.8669415025117403e-20]
  * Hf(x): [883.6474886384784,-419.8559629573329,-419.85596295733484,199.9412037196877]
  * Number of iterations: [9]
  * Number of function calls: [13]
  * Number of gradient calls: [10]
  * Converged: [true]


 julia> @btime soptimize(rosenbrock2, $res.minimizer, hguess = $res.h)
   519.843 ns (4 allocations: 256 bytes)
 Results of Static Optimization Algorithm
  * Initial guess: [0.9999999999990606,0.9999999999980389]
  * Minimizer: [1.0499999999999563,1.1024999999999325]
  * Minimum: [6.04912840227379e-26]
  * Hf(x): [884.500949849433,-420.2366662662608,-420.2366662663068,200.1118104053618]
  * Number of iterations: [9]
  * Number of function calls: [9]
  * Number of gradient calls: [9]
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
  438.444 ns (1 allocation: 80 bytes)
Results of Static Optimization Algorithm
 * Initial guess: [0.5]
 * Minimizer: [0.40000000000390024]
 * Minimum: [1.0216512475319812]
 * Hf(x): [5.555568057253705]
 * Number of iterations: [5]
 * Number of function calls: [7]
 * Number of gradient calls: [6]
 * Converged: [true]

julia> using Optim

julia> @btime optimize(f, 0.1, 0.9) # Maybe a better idea
  429.201 ns (2 allocations: 176 bytes)
Results of Optimization Algorithm
 * Algorithm: Brent's Method
 * Search Interval: [0.100000, 0.900000]
 * Minimizer: 4.000000e-01
 * Minimum: 1.021651e+00
 * Iterations: 8
 * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(1.5e-08*|x|+2.2e-16): true
 * Objective Function Calls: 9

 julia> @btime soptimize(f, 0.45, hguess = 5.5) # This converges very fast when the guess is close
  286.170 ns (2 allocations: 96 bytes)
Results of Static Optimization Algorithm
 * Initial guess: [0.45]
 * Minimizer: [0.399999999987932]
 * Minimum: [1.0216512475319814]
 * Hf(x): [5.555567000416141]
 * Number of iterations: [4]
 * Number of function calls: [4]
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
  161.271 ns (0 allocations: 0 bytes)
(x = -0.04081632661278052, fx = -4.027922440030807e-11, isroot = true, iter = 3)

julia> @btime sroot(f, (-0.5, 0.5))
  223.031 ns (0 allocations: 0 bytes)
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
   2.828 μs (4 allocations: 256 bytes)
 Results of Static Optimization Algorithm
  * Initial guess: [0.0,0.5,0.5]
  * Minimizer: [-0.046069913165463626,0.3907860173721792,0.4093061224594615]
  * Minimum: [4.359930872763706e-21]
  * Hf(x): [21.530366536879317,-46.71040662443704,52.76696109347479,-46.710406624437184,213.63034559461593,-8.50337391439737,52.76696109347513,-8.503373914398187,241.31135228009185]
  * Number of iterations: [14]
  * Number of function calls: [28]
  * Number of gradient calls: [16]
  * Converged: [true]
```
