include("aaaz.jl")

"""
    aaai(f; degree=150, lawson=0, tol=1e-13, mero=false, plots=false, numtype=Float64)
Construct a rational approximation for `f` on the imaginary axis or right half-plane.

# Outputs
- callable function for the rational approximation
- vector of poles
- vector of residues
- vector of zeros
- final error estimate
- interpolation nodes
# Examples
```juliarepl
julia> f = z -> 1 / sqrt(z+3);   # anaylytic in the right half-plane
julia> r,pol,_ = aaai(f);  pol[end]
-3.0684163566735116 - 0.0006971615051296282im

julia> f = z -> 1 / sqrt(Complex(3-z));   # singularity in RHP
julia> r,pol,_ = aaai(f,mero=true);  pol[1]
3.0686924924965138 - 0.0007562668474104505im
```
"""

function aaai(f; kwargs...)
    M = 1.207
    if haskey(kwargs, :numtype)
        M = kwargs[:numtype](M)              # change float type
    end
    
    # map from unit circle to imaginary axis
    function zw(w)
        # avoid 1 over complex 0, which is NaN:
        z = w==1 ? Inf : M*(1+w) / (1-w) 
        return abs(z) > 1e50 ? 1e50 : z      # prevent Inf
    end
    
    # map from imaginary axis to unit circle
    function wz(z)
        w = (z-M) / (z+M)
        return abs(w) > 1e50 ? 1e50im : w    # prevent Inf
    end

    fw = f ∘ zw                              # compose with change of vars
    r, pol, res, zer, err, S = aaaz(fw; kwargs...)
    return (r ∘ wz,                          # undo variable change
        zw.(pol),                            # transplant poles
        (@. 2M*res/(1-pol)^2),               # chain rule
        zw.(zer),                            # transplant zeros
        err, 
        1im*imag(zw.(S))                     # cleanly transplant nodes
    )
end