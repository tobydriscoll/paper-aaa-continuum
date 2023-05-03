# Julia codes for *AAA rational approximation on a continuum*

These codes go with the paper *AAA rational approximation on a continuum* by Driscoll, Nakatsukasa, and Trefethen, submitted to SISC in 2023.

## Usage

To use `aaax`, you need to first include its definition:
```julia
include("aaax.jl")
```
Other dependencies will be loaded automatically; Julia's standard library should be sufficient. After that, you can run `aaax` as follows:

```juliarepl
julia> r,_ = aaax(exp);
julia> r(0.33) - exp(0.33)
-2.220446049250313e-16

julia> r,_ = aaax(exp, degree=3);    # type (3,3) rational approximation
julia> r(0.33) - exp(0.33)
-3.8581892347622215e-8

julia> r,_ = aaax(exp, degree=3, lawson=20);    # take 20 lawson minimax iterations
```

The same procedure applies for `aaaz` and `aaai`, each of which is defined in its own source file. You can find some usage examples in the documentation strings.

## Convergence plots

If you have the [`Plots`](https://docs.juliaplots.org/latest/) package installed, you can also get a convergence plot of the AAA iteration by first invoking `using Plots` once and then adding `plots=true` to the input arguments of the AAA calls.

## Extended precision

Any numeric type for which `svd` and `eigvals` are defined can be used. The built-in `BigFloat` type should work if you install [`GenericLinearAlgebra`](https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl), or you can use [`DoubleFloats`](https://github.com/JuliaMath/DoubleFloats.jl) if you want only quad precision. For example,

```juliarepl
julia> import Pkg; Pkg.add("DoubleFloats")
julia> using DoubleFloats
julia> r,_ = aaax(exp, tol=1e-28, numtype=Double64);
julia> x = Double64(2)/10;
julia> r(x) - exp(x)
-7.703719777548943e-32
```

