using LinearAlgebra, Statistics, Printf
include("common.jl")

"""
    aaaz(f; degree=150, lawson=0, tol=1e-13, mero=false, plots=false, numtype=Float64)
Construct a rational approximation for `f` on the unit circle or disk.

# Outputs
- callable function for the rational approximation
- vector of poles
- vector of residues
- vector of zeros
- final error estimate
- interpolation nodes
# Examples
```juliarepl
julia> r,_ = aaaz(exp);
julia> r(0.25) - exp(0.25)
0.0 - 8.485559860589348e-17im

julia> r,_ = aaaz(exp, degree=3);    # type (3,3) rational approximation
julia> r(0.25) - exp(0.25)
-3.0309762144575814e-7 - 3.265443968399623e-7im

julia> r,_ = aaaz(exp, degree=3, lawson=20);    # take 20 lawson minimax iterations

julia> f = z -> tan(π*z);
julia> r,pol,res,zer,_ = aaaz(f, degree=7, mero=true);   # allow poles in the disk
julia> Dict(zip(pol,res))
Dict{ComplexF64, ComplexF64} with 7 entries:
  1.50129-0.000487375im  => -0.32216+0.00129332im
  -1.5012-3.0955e-5im    => -0.321915-0.000245695im
  2.91176-0.0689825im    => -0.816777+0.0743735im
  -2.88821-0.0383128im   => -0.787576-0.0480784im
  -18.4101-63.8583im     => 445.143-273.031im
  -0.500001+2.09311e-7im => -0.318309-4.25566e-7im
  0.499998+7.99326e-7im  => -0.318311+5.89933e-7im

julia> zer
7-element Vector{ComplexF64}:
      -5.38085754377836 - 0.2991043804689039im
     -2.041900579039462 - 0.003963550140797963im
    -0.9999999999999998 - 1.0166042923979753e-16im
 -2.2800123217229256e-6 + 7.766919669234381e-7im
     1.0000000000000004 - 1.2242286762492256e-16im
      2.044543220021608 - 0.010323526740179008im
        5.5609649800704 - 0.4259693058977332im
```
"""
function aaaz(f; 
    degree=150,                              # max degree 
    lawson=0,                                # number of Lawson iterations 
    numtype=Float64,                         # floating-point precision 
    tol=1000*eps(numtype),                   # relative stopping tolerance 
    mero=false,                              # disallow poles in the disk? 
    plots=false                              # display convergence plots?
    )
    
    S = numtype.([-1, 1])                    # vector of support points
    f0 = f.( [S; ZS(S,10)] )                 # check for constant function...
    err = std(f0)                            # ...or degree==0
    if iszero(err) || degree==0 || (abs(err/mean(f0)) <= tol) 
        return x -> mean(f0), [], [], [], 0, S
    end

    best = nothing                           # track the best so far
    err, nbad = [], []                       # convergence statistics
    while true                               # MAIN AAA LOOP
        m = length(S)
        Z = ZS(S, max(3, 16-m))              # vector of test points
        fZ, fS = f.(Z), f.(S)                # evaluate f
        C = [1/(z-s) for z in Z, s in S]     # Cauchy matrix
        A = [fz-fs for fz in fZ, fs in fS] .* C        # Loewner matrix
        _, _, V = svd(A)                     # SVD
        w = V[:,end]                         # barycentric weights
        R = (C * (w.*fS)) ./ (C * w)         # approximant at test points
        push!(err, norm(fZ - R, Inf))        # track max error
        pol = poles(S, fS, w)                # poles of this approximant
        bad = mero ? 
            falses(length(pol)) :            # anything goes, or...
            (abs.(pol) .<= 1)                # ...flag bad poles
        push!(nbad, count(bad))              # track number of bad poles
        fmax = max( norm(fS, Inf), norm(fZ, Inf) )     # set scale of f
        if isnothing(best) || 
            (!any(bad) && (last(err) < err[best.m-1]))
            best = (; m, S, w)               # save new best result
        end
        is_low = (err[best.m-1]/fmax < 1e-2) # has error decreased some?
        if (!any(bad) && (last(err)/fmax <= tol)) ||   # stop if converged
            (m == degree+1) ||                         # ...or at max degree
            ((m-best.m >= 10) && is_low)     # ...or if stagnated
            break
        end
        _, j = findmax(abs, fZ-R)            # find next support point...
        S = [S; Z[j]]                        # ...and include it
    end

    m, S, w = best
    fS = f.(S)
    r = evaluator(S, fS, w)                  # create callable function
    pol, res, zer = prz(S, fS, w)            # poles, residues, zeros

    if lawson > 0                            # LAWSON ITERATION
        Z = ZS(S, 20)                        # switch to finer grid
        fZ = f.(Z)                           # evaluate f
        wt = ones(numtype, m + length(Z))    # Lawson weight vector
        C = [1/(z-s) for z in Z, s in S]     # Cauchy matrix
        R = (C * (w.*fS)) ./ (C*w)           # approximant at test points
        F = [fZ; fS]                         # all values of f...
        fmax = norm(F,Inf)                   # scale of f
        R = [R; fS]                          # ...and its approximant
        A = [C; I]                           # numerator
        B = [fZ.*C; diagm(fS)] / fmax        # f * denominator
        a = []    # put into scope
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

    zz = ZS(S,30) 
    ee = f.(zz) - r.(zz)         # COMPUTE ERROR AND PLOT
    final_err = norm(ee, Inf)

    if plots 
        # this block requires Plots
        lay = reshape([(label=:a, blank=false), Plots.GridLayout(1, 2)], 2, 1)
        fig = plot(layout=lay, legend=false, size=(800, 800), titlefontsize=12)
        color = [ (k > 0 ? :red : :darkblue) for k in nbad ]
        scatter!( [m-1], [final_err], subplot=1, 
                  ms=7, color=RGBA(1,1,1,0), msw=3, msc=:orange )
        scatter!( err; color, subplot=1, msw=0, title="convergence",
                  xlabel="degree (=m-1)", yaxis=(:log10, "max error") )

        title = @sprintf "error: max = %.1e" final_err
        scl = 1.5*max(final_err, floatmin(numtype))
        color = lawson > 0 ? RGB(0.7,0,0.9) : :darkblue
        plot!(real(ee), imag(ee); color, subplot=2, 
                    xlims=[-scl,scl], ylims=[-scl,scl], aspect_ratio=1, title)
        if lawson > 0
            ΔΘ = diff(angle.(ee) / 2π)
            wind = round(Int, sum( mod(t+0.5, 1)-0.5 for t in ΔΘ ) )
            annotate!(0.95*scl, -0.83*scl, subplot=2,
                        ("winding num = $wind", 9, :right, :bottom) )
        end
                    
        scl = length(pol) > 0 ? 1.5 + minimum(abs.(pol)) : 2.5
        θ = range(0, 2π, 300)
        plot!(cos.(θ), sin.(θ), subplot=3, l=(2,:black), 
            xlims=[-scl,scl], ylims=[-scl,scl], aspect_ratio=1)
        scatter!(real(pol), imag(pol), subplot=3, msw=0, color=:darkred, title="poles")
        display(fig)
    end
    return r, pol, res, zer, final_err, S

end

"Create sample points on the unit circle"
function ZS(S, p)
    T = sort(angle.(S)) 
    d = eltype(S).(collect(1:p) / (p+1))     # fractional step sizes
    dT = diff( [T; T[1]+2π] )
    ZT = T' .+ d.*dT'
    return cis.(vec(ZT))
end
