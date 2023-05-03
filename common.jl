"""
    poles(z, y, w)
Compute poles and residues of a rational function in barycentric form.

`z` is a vector of interpolation nodes (i. e., support points), `y` is 
a vector of values at the nodes, and `w` is a vector of barycentric
weights. The outputs are vectors of poles and corresponding
residues.
"""
function poles(zj, fj, wj)
    m = length(wj) 
    if any(wj .== 0)
        ii = findall(wj .!= 0)
        m = length(ii)
        zj, fj, wj = zj[ii],  fj[ii],  wj[ii]
    end   
    B = diagm( [0; ones(m)] )
    E = [0 transpose(wj); ones(m) diagm(zj) ];
    pol = []  # put it into scope
    try 
        pol = filter( isfinite, eigvals(E, B) )
    catch
        # generalized eigen not available in extended precision, so:
        λ = eigvals(E\B)
        pol = filter( z->abs(z) < 1e14, 1 ./ λ)
    end
    return pol
end

"""
    prz(z, y, w)
Compute poles, residues, and zeros of a barycentric rational function.

`z` is a vector of interpolation nodes (i. e., support points), `y` is 
a vector of values at the nodes, and `w` is a vector of barycentric
weights. The outputs are a vector of poles, a vector ofcorresponding
residues, and a vector of roots.
"""
function prz(zj, fj, wj)       # POLES, RESIDUES, ZEROS
    pol = poles(zj, fj, wj)

    N = t -> sum( y*w / (t-x) for (x, y, w) in zip(zj, fj, wj) )
    Ddiff = t -> sum( -w / (t-x)^2 for (x, w) in zip(zj, wj) )
    res = similar(complex(pol))
    res .= N.(pol) ./ Ddiff.(pol)

    m = length(wj)
    E = [0 transpose(wj.*fj); ones(m) diagm(zj)]
    B = diagm( [0; ones(m)] )
    zer = []  # put it into scope
    try 
        zer = filter( isfinite, eigvals(E, B) )
    catch
        # generalized eigen not available in extended precision, so:
        λ = eigvals(E\B)
        zer = filter( z->abs(z) < 1e14, 1 ./ λ)
    end
    return pol, res, zer
end

"""
    evaluator(z, y, w)
Construct a rational function in barycentric form.

`z` is a vector of interpolation nodes (i. e., support points), `y` is 
a vector of values at the nodes, and `w` is a vector of barycentric
weights. The output is a callable function of one argument.
"""
function evaluator(zj, fj, wj)              # FUNCTION HANDLE FOR R
    w_times_f = wj.*fj
    return function(z)
        if isinf(z)
            return sum(w_times_f) / sum(wj)
        end
        k = findfirst(z .== zj)
        if isnothing(k)         # not at any node
            C = @. 1 / (z - zj)
            return sum(C.*w_times_f) / sum(C.*wj) 
        else                    # interpolation at node
            return fj[k]   
        end
    end  
end
