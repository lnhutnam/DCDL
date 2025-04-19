# DCDL.jl: Dictionary Learning with Difference of Convex Programming

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

```julia
juliausing DCDL

# Create solver with default parameters
solver = DCDLSolver()

# Generate data
generate_data!(solver)

# Solve the dictionary learning problem
solve!(solver)

# Plot results
plot_results(solver)
```