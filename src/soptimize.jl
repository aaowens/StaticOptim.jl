# Much of this code is lifted from LineSearches.jl
# I modified it to accept StaticArrays and not allocate

# Some of the optimization code is adapted from Optim.jl


@with_kw struct BackTracking{TF, TI}
    c_1::TF = 1e-4
    ρ_hi::TF = 0.5
    ρ_lo::TF = 0.1
    iterations::TI = 1_000
    maxstep::TF = Inf
end

abstract type BackTrackingOrder end
struct Order2 <: BackTrackingOrder end
struct Order3 <: BackTrackingOrder end
struct Order0 <: BackTrackingOrder end
ordernum(::Order2) = 2
ordernum(::Order3) = 3

struct StaticOptimizationResult{TS <: Union{SVector, Number}, TV <: Union{SMatrix, Number}}
    minimum::Float64
    minimizer::TS
    normjx::Float64
    iter::Int
    hx::TV
    converged::Bool
end

function soptimize(f, x::StaticVector, bto::BackTrackingOrder = Order3(), hguess = nothing)
    res = DiffResults.GradientResult(x)
    ls = BackTracking()
    order = ordernum(bto)
    tol = 1e-8
    x_new = copy(x)
    hx = diagm(ones(x))
    if !(hguess isa Nothing)
        hx = hguess * hx
    end
    hold = copy(hx)
    jold = copy(x); s = copy(x)
    @unpack c_1, ρ_hi, ρ_lo, iterations = ls
    iterfinitemax = -log2(eps(eltype(x)))
    sqrttol = sqrt(eps(Float64))
    α_0 = 1.
    N = 200
    for n = 1:N
        res = ForwardDiff.gradient!(res, f, x) # Obtain gradient
        ϕ_0 = DiffResults.value(res)
        isfinite(ϕ_0) || return StaticOptimizationResult(NaN, NaN*x, NaN, n, hx, false)
        jx = DiffResults.gradient(res)
        norm(jx, Inf) < tol && return StaticOptimizationResult(ϕ_0, x, norm(jx, Inf), n, hx, true)
        n == N && return StaticOptimizationResult(ϕ_0, x, norm(jx, Inf), n, hx, false)
        if n > 1 # update hessian
            y = jx - jold
            hx = norm(y) < eps(eltype(x)) ? hx : hx + y*y' / (y'*s) - (hx*(s*s')*hx)/(s'*hx*s)
        end
        s = -hx\jx # Obtain direction
        dϕ_0 = dot(jx, s)
        if dϕ_0 >= 0. # If bad, reset search direction
            hx = hold
            s = -jx
            dϕ_0 = dot(jx, s)
        end
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        ϕx_1 = f(x + α_1*s)

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            iterfinite += 1
            α_1 = α_2
            α_2 = α_1/2
            ϕx_1 = f(x + α_2*s)
        end

        # Backtrack until we satisfy sufficient decrease condition
        while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
            # Increment the number of steps we've had to perform
            iteration += 1

            # Ensure termination
            if iteration > iterations
                error("Linesearch failed to converge, reached maximum iterations $(iterations).",
                α_2)
            end

            # Shrink proposed step-size:
            if order == 2 || iteration == 1
                # backtracking via quadratic interpolation:
                # This interpolates the available data
                #    f(0), f'(0), f(α)
                # with a quadractic which is then minimised; this comes with a
                # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
                # provided that c_1 < 1/2; the backtrack_condition at the beginning
                # of the function guarantees at least a backtracking factor ρ.
                α_tmp = - (dϕ_0 * α_2^2) / ( 2 * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
            else
                div = 1. / (α_1^2 * α_2^2 * (α_2 - α_1))
                a = (α_1^2*(ϕx_1 - ϕ_0 - dϕ_0*α_2) - α_2^2*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div
                b = (-α_1^3*(ϕx_1 - ϕ_0 - dϕ_0*α_2) + α_2^3*(ϕx_0 - ϕ_0 - dϕ_0*α_1))*div

                if norm(a) <= eps(Float64) + sqrttol*norm(a)
                    α_tmp = dϕ_0 / (2*b)
                else
                    # discriminant
                    d = max(b^2 - 3*a*dϕ_0, 0.)
                    # quadratic equation root
                    α_tmp = (-b + sqrt(d)) / (3*a)
                end
            end
            α_1 = α_2

            α_tmp = NaNMath.min(α_tmp, α_2*ρ_hi) # avoid too small reductions
            α_2 = NaNMath.max(α_tmp, α_2*ρ_lo) # avoid too big reductions

            # Evaluate f(x) at proposed position
            ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s)
        end
        alpha, fpropose = α_2, ϕx_1

        s = alpha*s
        x = x + s # Update x
        jold = copy(jx)
    end
    return StaticOptimizationResult(NaN, NaN*x, NaN, N, hx, false)
end


function soptimize(f, x::Number, hguess = nothing)
    res = DiffResults.DiffResult(x, (x,))
    ls = BackTracking()
    tol = 1e-8
    x_new = copy(x)
    hx = one(x)
    hold = copy(hx)
    if !(hguess isa Nothing)
        hx = hguess*one(x)
    end
    jold = copy(x); s = copy(x)
    @unpack c_1, ρ_hi, ρ_lo, iterations = ls
    iterfinitemax = -log2(eps(eltype(x)))
    sqrttol = sqrt(eps(Float64))
    α_0 = 1.
    N = 200

    res = ForwardDiff.derivative!(res, f, x) # Obtain gradient
    ϕ_0 = DiffResults.value(res)
    isfinite(ϕ_0) || return StaticOptimizationResult(NaN, NaN*x, NaN, 1, hx, false)
    jx = DiffResults.derivative(res)
    norm(jx, Inf) < tol && return StaticOptimizationResult(ϕ_0, x, norm(jx, Inf), 1, hx, true)
    needsupdate = false
    for n = 1:N
        if needsupdate
            res = ForwardDiff.derivative!(res, f, x) # Obtain gradient
            needsupdate = false
        end
        ϕ_0 = DiffResults.value(res)
        isfinite(ϕ_0) || return StaticOptimizationResult(NaN, NaN*x, NaN, n, hx, false)
        jx = DiffResults.derivative(res)
        norm(jx, Inf) < tol && return StaticOptimizationResult(ϕ_0, x, norm(jx, Inf), n, hx, true)
        n == N && return StaticOptimizationResult(ϕ_0, x, norm(jx, Inf), n, hx, false)
        if n > 1 # update hessian
            y = jx - jold
            hx =  y / s
        end
        s = -hx\jx # Obtain direction
        dϕ_0 = dot(jx, s)
        if dϕ_0 >= 0. # If bad, reset search direction
            hx = hold
            s = -jx
            dϕ_0 = dot(jx, s)
        end
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        res = ForwardDiff.derivative!(res, f, x + α_1*s) # Obtain gradient

        ϕx_1 = DiffResults.value(res)

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            needsupdate = true
            iterfinite += 1
            α_1 = α_2
            α_2 = α_1/2
            ϕx_1 = f(x + α_2*s)
        end

        # Backtrack until we satisfy sufficient decrease condition
        while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
            needsupdate = true
            # Increment the number of steps we've had to perform
            iteration += 1

            # Ensure termination
            if iteration > iterations
                error("Linesearch failed to converge, reached maximum iterations $(iterations).",
                α_2)
            end

            # Shrink proposed step-size:

            # backtracking via quadratic interpolation:
            # This interpolates the available data
            #    f(0), f'(0), f(α)
            # with a quadractic which is then minimised; this comes with a
            # guaranteed backtracking factor 0.5 * (1-c_1)^{-1} which is < 1
            # provided that c_1 < 1/2; the backtrack_condition at the beginning
            # of the function guarantees at least a backtracking factor ρ.
            α_tmp = - (dϕ_0 * α_2^2) / ( 2 * (ϕx_1 - ϕ_0 - dϕ_0*α_2) )
            α_1 = α_2

            α_tmp = NaNMath.min(α_tmp, α_2*ρ_hi) # avoid too small reductions
            α_2 = NaNMath.max(α_tmp, α_2*ρ_lo) # avoid too big reductions

            # Evaluate f(x) at proposed position
            ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s)
        end
        alpha, fpropose = α_2, ϕx_1

        s = alpha*s
        x = x + s # Update x
        jold = copy(jx)
    end
    return StaticOptimizationResult(NaN, NaN*x, NaN, N, hx, false)
end

function soptimize(f, x::Number, bto::Order0, hguess = nothing)
    res = DiffResults.DiffResult(x, (x,))
    tol = 1e-8
    hx = one(x)
    hold = copy(hx)
    iterfinitemax = -log2(eps(eltype(x)))
    if !(hguess isa Nothing)
        hx = hguess*one(x)
    end
    jold = copy(x); s = copy(x)
    α_0 = 1.
    N = 200
    for n = 1:N
        res = ForwardDiff.derivative!(res, f, x) # Obtain gradient
        ϕ_0 = DiffResults.value(res)
        isfinite(ϕ_0) || return StaticOptimizationResult(NaN, NaN*x, NaN, n, hx, false)
        jx = DiffResults.derivative(res)
        norm(jx, Inf) < tol && return StaticOptimizationResult(ϕ_0, x, norm(jx, Inf), n, hx, true)
        n == N && return StaticOptimizationResult(ϕ_0, x, norm(jx, Inf), n, hx, false)
        if n > 1 # update hessian
            y = jx - jold
            hx = abs(y) < eps(eltype(x)) ? hx : y / s
        end
        s = -jx/hx # Obtain direction
        #### Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        ϕx_1 = f(x + α_1*s)

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            iterfinite += 1
            α_1 = α_2
            α_2 = α_1/2
            ϕx_1 = f(x + α_2*s)
        end
        alpha = α_2

        s = alpha*s
        x = x + s # Update x
        jold = jx
    end
    return StaticOptimizationResult(NaN, NaN*x, NaN, N, hx, false)
end


function Base.show(io::IO, r::StaticOptimizationResult)
    @printf io "Results of Static Optimization Algorithm\n"
    @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
    @printf io " * Minimum: [%s]\n" join(r.minimum, ",")
    @printf io " * |Df(x)|: [%s]\n" join(r.normjx, ",")
    @printf io " * Hf(x): [%s]\n" join(r.hx, ",")
    @printf io " * Number of iterations: [%s]\n" join(r.iter, ",")
    @printf io " * Converged: [%s]\n" join(r.converged, ",")
    return
end


### Modified Newton algorithm
# I made this up, I do not know if it has nice convergence properties
# It first computes the value and derivative and performs a Newton step
# Then it computes the value and the potentially unnecessary derivative at the Newton point
# If this an improvement (closer to 0), the Newton step is accepted and it performs another Newton step
# If this is worse, it backtracks if the candidate is not finite
# Then it performs a Halley step using the function value at the candidate point
# If this is an improvement, the Halley step is accepted
# If it's still not an improvement, simple backtracking is done until it is
function snewton(f, x::Number)
    x = float(x)
    res = DiffResults.DiffResult(x, (x,))
    tol = 1e-8
    iterfinitemax = -log2(eps(eltype(x)))
    α_0 = 1.
    N = 200
    res = ForwardDiff.derivative!(res, f, x) # Obtain gradient
    ϕ_0 = DiffResults.value(res)
    abs(ϕ_0) < tol && return (x = x, fx = ϕ_0)
    isfinite(ϕ_0) || return (x = NaN*x, fx = NaN)

    needsupdate = false
    for n = 1:N
        if needsupdate
            res = ForwardDiff.derivative!(res, f, x) # Obtain gradient
            needsupdate = false
        end
        ϕ_0 = DiffResults.value(res)
        abs(ϕ_0) < tol && return (x = x, fx = ϕ_0)
        isfinite(ϕ_0) || return (x = NaN*x, fx = NaN)
        jx = DiffResults.derivative(res)
        x2 = x - ϕ_0/jx

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        res = ForwardDiff.derivative!(res, f, x2) # Obtain gradient
        ϕx_1 = DiffResults.value(res)
        abs(ϕx_1) < tol && return (x = x2, fx = ϕx_1)

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        x2old = x2
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            needsupdate = true
            iterfinite += 1
            α_1 = α_2
            α_2 = α_1/2
            x2 = (1 - α_2)*x + α_2*x2old # convex combination
            ϕx_1 = f(x2)
        end

        α_h = 1.
        # Backtrack until we satisfy sufficient decrease condition
        if abs(ϕx_1) > abs(ϕx_0) # Closer to 0?
            needsupdate = true
            # Increment the number of steps we've had to perform
            # Interpolate available data using quadratic
            hx = 2(ϕx_1 - ϕx_0 - jx*(x2 - x) ) / (x2 - x)^2 # Taylor approximation
            x2 = x - 2ϕx_0*jx / (2jx^2 - ϕx_0*hx) # Halley step
            ϕx_1 = f(x2)
            iteration = 0
            # If still not closer to 0, do simple backtracking until we are
            while abs(ϕx_1) > abs(ϕx_0)
                iteration += 1
                α_1 = α_h
                α_2 = α_1/2
                x2 = (1 - α_2)*x + α_2*x2 # convex combination
                ϕx_1 = f(x2)
                iteration > 30 && error("Failed to converge")
            end
        end
        x = x2
    end
    return (x = NaN*x, fx = NaN)
end

### Copied from Tamas Papp on Discourse

"""
    bisection(f, a, b; fa = f(a), fb = f(b), ftol, wtol)

Bisection algorithm for finding the root ``f(x) ≈ 0`` within the initial bracket
`[a,b]`.

Returns a named tuple

`(x = x, fx = f(x), isroot = ::Bool, iter = ::Int, ismaxiter = ::Bool)`.

Terminates when either

1. `abs(f(x)) < ftol` (`isroot = true`),
2. the width of the bracket is `≤wtol` (`isroot = false`),
3. `maxiter` number of iterations is reached. (`isroot = false, maxiter = true`).

which are tested for in the above order. Therefore, care should be taken not to make `wtol` too large.

"""
function bisection(f, a::Real, b::Real; fa::Real = f(a), fb::Real = f(b),
                   ftol = √eps(), wtol = 0, maxiter = 100)
                   a, b = float(a), float(b)
    @assert fa * fb ≤ 0 "initial values don't bracket zero"
    @assert isfinite(a) && isfinite(b)
    _bisection(f, float.(promote(a, b, fa, fb, ftol, wtol))..., maxiter)
end

function _bisection(f, a, b, fa, fb, ftol, wtol, maxiter)
    iter = 0
    abs(fa) < ftol && return (x = a, fx = fa, isroot = true, iter = iter, ismaxiter = false)
    abs(fb) < ftol && return (x = b, fx = fb, isroot = true, iter = iter, ismaxiter = false)
    while true
        iter += 1
        m = middle(a, b)
        fm = f(m)
        abs(fm) < ftol && return (x = m, fx = fm, isroot = true, iter = iter, ismaxiter = false)
        abs(b-a) ≤ wtol && return (x = m, fx = fm, isroot = false, iter = iter, ismaxiter = false)
        if fa * fm > 0
            a, fa = m, fm
        else
            b, fb = m, fm
        end
        iter == maxiter && return (x = m, fx = fm, isroot = false, iter = iter, ismaxiter = true)
    end
end


sroot(f, x::Number) = snewton(f, x)
sroot(f, x::Tuple{T, T}) where T <: Number = bisection(f, x[1], x[2])
