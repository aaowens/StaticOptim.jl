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
struct StaticOptimizationResults{Tx, Th, Tf}
    initial_x::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int
    g_converged::Bool
    g_tol::Tf
    f_calls::Int
    g_calls::Int
    h::Th
end

setresult(x::StaticVector) = DiffResults.GradientResult(x)
setresult(x::Number) = DiffResults.DiffResult(x, x)
initialh(x::StaticVector{P,T}) where {P,T} = SMatrix{P,P,T}(I)
initialh(x::Number) = one(x)

setgradient!(res, f, x::StaticVector) = ForwardDiff.gradient!(res, f, x)
setgradient!(res, f, x::Number) = ForwardDiff.derivative!(res, f, x)
function getgradient(res::DiffResults.ImmutableDiffResult{1,N,Tuple{T}}) where {N <: Number, T <: StaticVector}
    DiffResults.gradient(res)
end
function getgradient(res::DiffResults.ImmutableDiffResult{1,T,Tuple{T}}) where T <: Number
    DiffResults.derivative(res)
end

function soptimize(f, x::Union{StaticVector{P,T}, TN}, bto::BackTrackingOrder = Order2();
    hguess = nothing, tol = 1e-8,
    updating = true, maxiter = 200) where {P,T, TN <: Number}
    res = setresult(x)
    ls = BackTracking()
    order = ordernum(bto)
    xinit = copy(x)
    x_new = copy(x)
    if hguess !== nothing
        hx = hguess
    else
        hx = initialh(x)
    end
    hold = copy(hx)
    jold = copy(x); s = copy(x)
    @unpack c_1, ρ_hi, ρ_lo, iterations = ls
    iterfinitemax = -log2(eps(eltype(x)))
    sqrttol = sqrt(eps(Float64))
    α_0 = 1.
    f_calls = 0
    g_calls = 0
    ## When updating is false, the initial linesearch check computes just
    # the value and the updating logic is skipped everywheres.

    ## When updating is true, the algorithm computes both the value
    # and the gradient in the initial linesearch check
    # In cases where the initial guess is usually accepted, this is
    # more efficient because we get the value for free with the gradient,
    # so we skip a redundent function call.

    # For the first iteration no gradient is available, so update
    if updating
        needsupdate = true
    end
    for n = 1:maxiter
        ## Compute the gradient if needed
        if updating
            # If we didn't accept the first linesearch guess, the gradient
            # is outdated, so update
            if needsupdate
                res = setgradient!(res, f, x); f_calls +=1; g_calls +=1; # Obtain gradient
                needsupdate = false
            end
            # Otherwise do nothing
        else
            res = setgradient!(res, f, x); f_calls +=1; g_calls +=1; # Obtain gradient
        end
        ϕ_0 = DiffResults.value(res)
        ## Check convergence
        isfinite(ϕ_0) || return StaticOptimizationResults(xinit, NaN*x,
        NaN, n, false, tol, f_calls, g_calls, hx)
        jx = getgradient(res)
        norm(jx, Inf) < tol && return StaticOptimizationResults(xinit, x,
        ϕ_0, n, true, tol, f_calls, g_calls, hx)
        ## Update hessian if two gradients available
        if n > 1
            y = jx - jold
            hx = hx + y*y' / (y'*s) - (hx*(s*s')*hx)/(s'*hx*s)
        end
        ## Compute search directions
        s = -hx\jx
        dϕ_0 = dot(jx, s)
        if dϕ_0 >= 0. # If bad, reset search direction
            hx = hold
            s = -jx
            dϕ_0 = dot(jx, s)
        end
        ## Perform line search

        # Count the total number of iterations
        iteration = 0
        ϕx_0, ϕx_1 = ϕ_0, ϕ_0
        α_1, α_2 = α_0, α_0
        if updating
            res = setgradient!(res, f, x + α_1*s); f_calls +=1; g_calls +=1; # Obtain gradient
            ϕx_1 = DiffResults.value(res)
        else
            ϕx_1 = f(x + α_1*s); f_calls +=1;
        end

        # Hard-coded backtrack until we find a finite function value
        iterfinite = 0
        while !isfinite(ϕx_1) && iterfinite < iterfinitemax
            if updating
                needsupdate = true
            end
            iterfinite += 1
            α_1 = α_2
            α_2 = α_1/2
            ϕx_1 = f(x + α_2*s); f_calls += 1;
        end

        # Backtrack until we satisfy sufficient decrease condition
        while ϕx_1 > ϕ_0 + c_1 * α_2 * dϕ_0
            # If this part is reached we did not accept the initial linesearch
            # guess, so we will need to update on the next iterations
            if updating
                needsupdate = true
            end
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
            ϕx_0, ϕx_1 = ϕx_1, f(x + α_2*s); f_calls += 1;
        end
        alpha, fpropose = α_2, ϕx_1

        s = alpha*s
        x = x + s # Update x
        jold = copy(jx)
    end
    return StaticOptimizationResults(xinit, NaN*x,
    NaN, maxiter, false, tol, f_calls, g_calls, hx)
end

function Base.show(io::IO, r::StaticOptimizationResults)
    @printf io "Results of Static Optimization Algorithm\n"
    @printf io " * Initial guess: [%s]\n" join(r.initial_x, ",")
    @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
    @printf io " * Minimum: [%s]\n" join(r.minimum, ",")
    @printf io " * Hf(x): [%s]\n" join(r.h, ",")
    @printf io " * Number of iterations: [%s]\n" join(r.iterations, ",")
    @printf io " * Number of function calls: [%s]\n" join(r.f_calls, ",")
    @printf io " * Number of gradient calls: [%s]\n" join(r.g_calls, ",")
    @printf io " * Converged: [%s]\n" join(r.g_converged, ",")
    return
end


### Modified Newton algorithm for root-finding
# I made this up, I do not know if it has nice convergence properties
# It first computes the value and derivative and performs a Newton step
# Then it computes the value and the potentially unnecessary derivative at the Newton point
# If this an improvement (closer to 0), the Newton step is accepted and it performs another Newton step
# If this is worse, it backtracks if the candidate is not finite
# Then it performs a Halley step using the function value at the candidate point
# If this is an improvement, the Halley step is accepted
# If it's still not an improvement, simple backtracking is done until it is
function snewton(f, x::Number; maxiter = 200, tol = 1e-8)
    x = float(x)
    res = DiffResults.DiffResult(x, (x,))
    iterfinitemax = -log2(eps(eltype(x)))
    α_0 = 1.
    res = ForwardDiff.derivative!(res, f, x) # Obtain gradient
    ϕ_0 = DiffResults.value(res)
    abs(ϕ_0) < tol && return (x = x, fx = ϕ_0)
    isfinite(ϕ_0) || return (x = NaN*x, fx = NaN)

    needsupdate = false
    for n = 1:maxiter
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
function bisection(f, a::Real, b::Real;
                   ftol = √eps(), wtol = 0., maxiter = 100)
    a, b = float(a), float(b)
    fa, fb = f(a), f(b)
    fa * fb ≤ 0 || error("Not a bracket")
    (isfinite(a) && isfinite(b)) || error("Not finite")
    _bisection(f, a, b, fa, fb, ftol, wtol, maxiter)
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
sroot(f, x::Tuple{T, T}) where T <: Number = @inbounds bisection(f, x[1], x[2])
function sroot(f, x::SVector; hguess = nothing,
    updating = true)
    f2(s) = sum(f(s).^2)
    soptimize(f2, x, hguess = hguess, updating = updating )
end
