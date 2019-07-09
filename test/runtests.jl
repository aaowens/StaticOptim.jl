using StaticOptim, Optim
using Test

# All of these tests are from OptimTestProblems.jl

using StaticArrays
sx = @SVector ones(2)
sx = 3.2 * sx
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
res = soptimize(rosenbrock, sx)
@test res.g_converged == true
@test rosenbrock(res.minimizer) == res.minimum
@test show(res) === nothing
res = soptimize(rosenbrock, sx)
@test res.g_converged == true
res = soptimize(rosenbrock, sx, updating = true)
@test res.g_converged == true
res = soptimize(rosenbrock, sx, bto = StaticOptim.Order3())
@test res.g_converged == true
@test rosenbrock(res.minimizer) == res.minimum

res = soptimize(rosenbrock, sx/2)
@test res.g_converged == true
@test rosenbrock(res.minimizer) == res.minimum


# Constrained

function U(h, w)
    h1, h2 = h[1], h[2]
    h1 >= 1 && return -Inf*one(h1)
    h2 >= 1 && return -Inf*one(h1)
    c = 2w*h1 + 1.5w*h2 + 0.1
    c <= 0 && return -Inf*one(h1)
    log(c) + log(1 - h1) + log(1 - h2)
end

l = SVector{2}(0., 0.4)
sx = SVector{2}(0.7, 0.7)
res = constrained_soptimize(x -> -U(x, 1), sx, lower = l)
@test res.g_converged == true
lbs = [SVector{2}(x, y) for x in 0:0.1:0.4, y in 0:0.1:0.4]
ubs = [SVector{2}(x, y) for x in 0.6:.1:0.9, y in 0.6:0.1:0.9]
ws = 0.7:0.1:1.01
sx = SVector{2}(0.5, 0.5)
constrained_soptimize(x -> -U(x, 1.), sx, lower = SVector{2}(0., 0.1))
sols = [(w = w, l = l, res = constrained_soptimize(x -> -U(x, w), sx, lower = l)) for l in lbs, u in ubs, w in ws]
@test all(sol.res.g_converged == true for sol in sols)

## Constrained, 4 people

function U(h, w)
    h1, h2, h3, h4 = h[1], h[2], h[3], h[4]
    h1 >= 1 && return -Inf*one(h1)
    h2 >= 1 && return -Inf*one(h1)
    h3 >= 1 && return -Inf*one(h1)
    h4 >= 1 && return -Inf*one(h1)
    c = 2w*h1 + 1.5w*h2 + 1.2w*h3 + 0.5w*h4 + 0.1
    c <= 0 && return -Inf*one(h1)
    log(c) + log(1 - h1) + log(1 - h2) + log(1 - h3) + log(1 - h4)
end

l = SVector{4}(0., 0., 0., 0.)
sx = SVector{4}(0.7, 0.7, 0.7, 0.7)
res = constrained_soptimize(x -> -U(x, 1), sx, lower = l)
@test res.g_converged == true
lbs = [SVector{4}(x, y, z, t) for x in 0:0.1:0.4, y in 0:0.1:0.4, z in 0:0.1:0.4, t in 0:0.1:0.4]
ubs = [SVector{4}(x, y, z, t) for x in 0.6:.1:0.9, y in 0.6:0.1:0.9, z in 0.6:0.1:0.9, t in 0.6:0.1:0.9]
ws = 0.5:0.5:2.
sx = SVector{4}(0.5, 0.5, 0.5, 0.5)
constrained_soptimize(x -> -U(x, 1), sx, lower = SVector{4}(0.3, 0., 0.2, 0.2))
#sols = [(w = w, l = l, res = constrained_soptimize(x -> -U(x, w), sx, lower = l, upper = u)) for l in lbs, u in ubs, w in ws];
@test all(constrained_soptimize(x -> -U(x, w), sx, lower = l, upper = u).g_converged for l in lbs, u in ubs, w in ws)
#sols = [(w = 1, l = l, res = constrained_soptimize(x -> -U(x, 1), sx, lower = l)) for l in lbs]


function fletcher_powell(x::AbstractVector)
    function theta(x::AbstractVector)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    return 100.0 * ((x[3] - 10.0 * theta(x))^2 +
        (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2
end

sx = @SVector ones(3)
sx = 3.2 * sx
res = soptimize(fletcher_powell, sx)
@test res.g_converged == true
@test fletcher_powell(res.minimizer) == res.minimum
res = soptimize(fletcher_powell, sx/2)
@test res.g_converged == true
@test fletcher_powell(res.minimizer) == res.minimum

function himmelblau(x::AbstractVector)
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end
sx = @SVector ones(2)
sx = 3.2 * sx
res = soptimize(himmelblau, sx)
@test res.g_converged == true
@test himmelblau(res.minimizer) == res.minimum
res = soptimize(himmelblau, sx/2)
@test res.g_converged == true
@test himmelblau(res.minimizer) == res.minimum

function powell(x::AbstractVector)
    return (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
        (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4
end
sx = @SVector ones(4)
sx = 3.2 * sx
res = soptimize(powell, sx)
@test res.g_converged == true
@test powell(res.minimizer) == res.minimum
res = soptimize(powell, sx/2)
@test res.g_converged == true
@test powell(res.minimizer) == res.minimum


### Univariate tests

f(x) = x^2 + 2*x
res = soptimize(f, 1.)
@test res.g_converged == true
@test f(res.minimizer) == res.minimum

function U(x)
    x = max(0., x)
    log(x)
end
f(h) = -(U(0.2 + h) + U(1 - h))
res = soptimize(f, 0.9)
@test res.g_converged == true
@test f(res.minimizer) == res.minimum

### Root finding
up(c) = c <= 0 ? Inf*c : 1/c
f(a) = up(2 - a) - .96up(2 + a)
out = sroot(f, 0.5)
@test out.fx < 1e-8
@test f(out.x) == out.fx

out = sroot(f, (-0.5, 0.5))
@test  out.isroot == true
@test f(out.x) == out.fx


const w = 3.
const beta = 0.96
const R = 1.01
const alpha_h = 1.5
uc(c) = 1/c
uh(h) = alpha_h/(1 - h)
function eulerfun(x)
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
x = SVector(0., 0.5, 0.5)
res = sroot(eulerfun, x)
@test res.g_converged == true

# Test regular arrays (Nonlinear least squares)
using Random
Random.seed!(1234)
realparam = rand(50)
const data = rand(5000, 50)
data[:, 1] = ones(5000)
const y = data*realparam .+ data*exp.(realparam) .+ randn(5000)
paramg = rand(50)
function obj(param)
yhat = data * param .+ data*exp.(param)
sum((yy - yh)^2 for (yy, yh) in zip(y, yhat))
end
sres = soptimize(obj, paramg);
@test res.g_converged == true
res = optimize(obj, paramg, method = Newton(), autodiff = :forward)
@test all(x -> abs(x) < 1e-6, res.minimizer - sres.minimizer)
l = rand(50)
u = rand(50)
u = [uu < ll ? Inf : uu for (uu, ll) in zip(u, l)]
m = (u + l)/2
cparamg = [isfinite(mm) ? mm : 2ll for (mm, ll) in zip(m, l)]
df = TwiceDifferentiable(obj, cparamg, autodiff = :forward)
dfc = TwiceDifferentiableConstraints(l, u)
resoc = optimize(df, dfc, cparamg, IPNewton())
sresoc = constrained_soptimize(obj, cparamg, lower = l, upper = u)

@test all(x -> abs(x) < 1e-6, resoc.minimizer - sresoc.minimizer)
