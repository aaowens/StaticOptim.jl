# StaticOptim

[![Build Status](https://travis-ci.org/aaowens/StaticOptim.jl.svg?branch=master)](https://travis-ci.org/aaowens/StaticOptim.jl)

[![Coverage Status](https://coveralls.io/repos/aaowens/StaticOptim.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/aaowens/StaticOptim.jl?branch=master)

[![codecov.io](http://codecov.io/github/aaowens/StaticOptim.jl/coverage.svg?branch=master)](http://codecov.io/github/aaowens/StaticOptim.jl?branch=master)

# Example:
```
using StaticOptim
using StaticArrays
sx = @SVector ones(2)
sx = 3.2 * sx
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

julia> res = soptimize(rosenbrock, sx)
StaticOptim.StaticOptimizationResult{StaticArrays.SArray{Tuple{2},Float64,1,2}}(3.7477721082170814e-24, [1.0, 1.0], 8.045984853069932e-11, 58)

```


