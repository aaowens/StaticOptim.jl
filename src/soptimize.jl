# Much of this code is lifted from LineSearches.jl
# I modified it to accept StaticArrays and not allocate


@with_kw struct BackTracking{TF, TI}
    c_1::TF = 1e-4
    ρ_hi::TF = 0.5
    ρ_lo::TF = 0.1
    iterations::TI = 1_000
    order::TI = 3
    maxstep::TF = Inf
end

struct StaticOptimizationResult{TS <: SVector}
    minimum::Float64
    minimizer::TS
    normjx::Float64
    iter::Int
end

function soptimize(f, x::StaticVector)
    res = DiffBase.GradientResult(x)
    ls = BackTracking()
    tol = 1e-8
    x_new = copy(x)
    hx = diagm(ones(x))
    jold = copy(x); s = copy(x)
    @unpack c_1, ρ_hi, ρ_lo, iterations, order = ls
    iterfinitemax = -log2(eps(eltype(x)))
    α_0 = 1.
    N = 100
    for n = 1:N
        res = ForwardDiff.gradient!(res, f, x) # Obtain gradient
        ϕ_0 = DiffBase.value(res)
        isfinite(ϕ_0) || return StaticOptimizationResult(NaN, NaN*x, NaN, n)
        jx = DiffBase.gradient(res)
        norm(jx) < tol && return StaticOptimizationResult(ϕ_0, x, norm(jx), n)
        if n > 1 # update hessian
            y = jx - jold
            ForwardDiff.hessian(f, x)
            hx = hx + y*y' / (y'*s) - (hx*(s*s')*hx')/(s'*hx*s)
        end
        s = -hx\jx # Obtain direction
        dϕ_0 = dot(jx, s)

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

                if norm(a) <= eps(Float64)
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
    return StaticOptimizationResult(NaN, NaN*x, NaN, N)
end
