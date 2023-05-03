using LinearAlgebra, Statistics, Printf
include("common.jl")

"""
    aaax(f; degree=150, lawson=0, tol=1e-13, plots=false, numtype=Float64)
Construct a rational approximation for `f` over the interval [-1,1].

# Outputs
- callable function for the rational approximation
- vector of poles
- vector of residues
- vector of zeros
- final error estimate
- interpolation nodes
# Examples
```juliarepl
julia> r,_ = aaax(exp);
julia> r(0.33) - exp(0.33)
-2.220446049250313e-16

julia> r,_ = aaax(exp, degree=3);    # type (3,3) rational approximation
julia> r(0.33) - exp(0.33)
-3.8581892347622215e-8

julia> r,_ = aaax(exp, degree=3, lawson=20);    # take 20 lawson minimax iterations
```
"""
function aaax(f; 
    degree=150,                              # max degree
    lawson=0,                                # number of Lawson iterations
    numtype=Float64,                         # floating-point precision
    tol=1000*eps(numtype),                   # relative stopping tolerance
    plots=false                              # display convergence plots?
    )

    S = numtype.([-1, 1])                    # vector of support points
    
    f0 = f.( [S; XS(S,10)] )                 # check for constant function...
    err = std(f0)                            # ...or degree==0
    if iszero(err) || degree==0 || (abs(err/mean(f0)) <= tol) 
        return x -> mean(f0), [], [], [], 0, S
    end 

    best = nothing                           # track the best so far
    err, nbad = [], []                       # convergence statistics
    while true                               # MAIN AAA LOOP
        m = length(S)
        X = XS(S, max(3, 16-m))              # vector of test points
        fX, fS = f.(X), f.(S)                # evaluate f
        C = [1/(x-s) for x in X, s in S]     # Cauchy matrix
        A = [fx-fs for fx in fX, fs in fS] .* C        # Loewner matrix
        _, _, V = svd(A)                     # SVD
        w = V[:,end]                         # barycentric weights
        R = (C * (w.*fS)) ./ (C * w)         # approximant at test points
        push!(err, norm(fX - R, Inf))        # track max error
        pol = poles(S, fS, w)                # poles of this approximant
        bad = @. (imag(pol) == 0) & (abs(pol) <= 1)    # flag bad poles
        push!(nbad, count(bad))              # track number of bad poles
        fmax = max( norm(fS, Inf), norm(fX, Inf) )     # set scale of f
        if isnothing(best) ||            
            (!any(bad) && (last(err) < err[best.m-1]))
            best = (; m, S, w)               # save new best result
        end
        is_low = (err[best.m-1]/fmax < 1e-2)
        if (!any(bad) && (last(err)/fmax <= tol)) ||   # stop if converged
            (m == degree+1) ||                         # ...or at max degree
            ((m-best.m >= 10) && is_low)     # ...or if stagnated
            break
        end
        _, j = findmax(abs, fX-R)            # find next support point...
        S = [S; X[j]]                        # ...and include it
    end

    m, S, w = best;  fS = f.(S)
    r = evaluator(S, fS, w)                  # create callable function
    pol, res, zer = prz(S, fS, w)            # poles, residues, zeros

    if lawson > 0                            # LAWSON ITERATION
        X = XS(S,20)                         # switch to finer grid
        fX = f.(X)                           # evaluate f
        wt = ones(numtype, m + length(X))    # Lawson weight vector
        C = [1/(x-s) for x in X, s in S]     # Cauchy matrix
        R = (C * (w.*fS)) ./ (C*w)           # approximant at test points
        F = [fX; fS]                         # all values of f...
        fmax = norm(F,Inf)                   # scale of f
        R = [R; fS]                          # ...and its approximant
        A = [C; I]                           # numerator
        B = [fX.*C; diagm(fS)] / fmax        # f * denominator
        a = []                               # put a into scope
        for _ in 1:lawson                    # take Lawson steps
            _, _, V = svd( sqrt.(wt).*[A B] )
            c = V[:,2m]
            a = c[1:m]                       # numerator weights
            w = -c[m+1:2m] / fmax            # denominator weights
            R = (A*a) ./ (A*w)               # approximant values
            wt = normalize( wt.*abs.(F-R), Inf )       # reweighting
        end
        r = evaluator(S, a./w, w)            # create callable function
    end

    xx = XS(S,30);  ee = f.(xx) - r.(xx)     # COMPUTE ERROR AND PLOT
    final_err = norm(ee, Inf)

    if plots 
        # this block requires Plots
        fig = plot(layout=(2,1), legend=false, size=(800, 800), titlefontsize=12)
        color = [ (k > 0 ? :red : :darkblue) for k in nbad ]
        scatter!( [m-1], [final_err], subplot=1, 
                  ms=7, color=RGBA(1,1,1,0), msw=3, msc=:orange )
        scatter!( err; color, subplot=1, msw=0, title="convergence",
                  xlabel="degree (=m-1)", yaxis=(:log10, "max error") )
        title = @sprintf "error: max = %.1e" final_err
        plot!(xx, ee; subplot=2, xlabel="x", title)
        display(fig)
    end

    return r, pol, res, zer, final_err, S
end

"Create sample points in [-1,1]"
function XS(S, p)
    S = sort(S) 
    d = eltype(S).(collect(1:p) / (p+1))     # fractional step sizes
    return vec( S[1:end-1]' .+ d.*diff(S)' )
end
