module DCDL

using LinearAlgebra
using Plots
using Random
using ArgParse

using Statistics

"""
Safety function to ensure all columns are normalized.

Args:
- A: Input matrix

Returns:
- Matrix with normalized columns
"""
function normalize_columns!(A::Matrix{Float64})
    for j in 1:size(A, 2)
        col = view(A, :, j)
        norm_val = norm(col)
        if norm_val > 0
            col ./= norm_val
        end
    end
    return A
end

"""
Calculate the recovery ratio between two dictionaries.

Args:
- original: Original dictionary
- new: New dictionary
- threshold: Distance threshold (default: 0.01)

Returns:
- Recovery ratio percentage
"""
function dictionaries_distance(
    original::AbstractMatrix, 
    new::AbstractMatrix, 
    threshold::Float64 = 0.01
)
    # Ensure dimensions match
    @assert size(original, 2) == size(new, 2) "Dictionary columns must match"
    
    distances = abs.(original' * new)
    counter = 0
    
    for i in 1:size(original, 2)
        min_value = 1 - maximum(distances[i, :])
        counter += min_value < threshold
    end
    
    return 100.0 * counter / size(original, 2)
end

"""
Dictionary Learning with Difference of Convex (DCDL) algorithm.
"""
mutable struct DCDLSolver
    # Algorithm parameters
    λ::Float64           # Penalty parameter
    γ::Float64           # Penalty function parameter
    max_iter::Int        # Maximum iterations
    length_gain::Int     # Length gain factor
    snr::Float64         # Signal-to-noise ratio
    inner_iter_max::Int  # Maximum inner iterations for dictionary update
    tolerance::Float64   # Convergence tolerance
    epsilon::Float64     # Small numerical constant
    
    # Data generation parameters
    dict_rows::Int       # Number of dictionary rows
    dict_cols::Int       # Number of dictionary columns
    cardinality::Int     # Number of non-zero coefficients
    data_length::Int     # Length of generated data
    
    # Result tracking
    recovery_ratio::Vector{Float64}
    total_error::Vector{Float64}
    
    # Learned components
    dictionary_true::Matrix{Float64}
    dictionary::Matrix{Float64}
    x_coef_matrix::Matrix{Float64}
    y_data_matrix::Matrix{Float64}
    
    # Constructor with default parameters
    function DCDLSolver(;
        λ::Float64 = 0.1, 
        γ::Float64 = 50000.0, 
        max_iter::Int = 1000, 
        length_gain::Int = 30, 
        snr::Float64 = 20.0,
        inner_iter_max::Int = 1,
        tolerance::Float64 = 1e-5,
        epsilon::Float64 = 1e-6,
        dict_rows::Int = 30,
        dict_cols::Int = 50,
        cardinality::Int = 1
    )
        # Calculate data length
        data_length = dict_cols * length_gain
        
        # Initialize result tracking arrays
        recovery_ratio = zeros(max_iter)
        total_error = zeros(max_iter)
        
        # Placeholders for dictionaries and matrices
        dictionary_true = zeros(dict_rows, dict_cols)
        dictionary = zeros(dict_rows, dict_cols)
        x_coef_matrix = zeros(dict_cols, data_length)
        y_data_matrix = zeros(dict_rows, data_length)
        
        new(
            λ, γ, max_iter, length_gain, snr, inner_iter_max, 
            tolerance, epsilon, dict_rows, dict_cols, cardinality, data_length,
            recovery_ratio, total_error,
            dictionary_true, dictionary, 
            x_coef_matrix, y_data_matrix
        )
    end
end

"""
Generate ground truth dictionary and data.
"""
function generate_data!(solver::DCDLSolver)
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Generate ground truth dictionary
    solver.dictionary_true = randn(solver.dict_rows, solver.dict_cols)
    normalize_columns!(solver.dictionary_true)
    
    # Generate coefficient matrix
    solver.x_coef_matrix = zeros(solver.dict_cols, solver.data_length)
    solver.x_coef_matrix[1:solver.cardinality, :] = randn(solver.cardinality, solver.data_length)
    
    # Randomize coefficient matrix
    for i in 1:solver.data_length
        solver.x_coef_matrix[:, i] = solver.x_coef_matrix[randperm(solver.dict_cols), i]
    end
    
    # Generate data matrix
    solver.y_data_matrix = solver.dictionary_true * solver.x_coef_matrix
    
    # Add noise if SNR is finite
    if solver.snr != Inf
        # Compute noise standard deviation
        data_std = std(vec(solver.y_data_matrix))
        noise_std = data_std * 10^(-solver.snr/20)
        
        # Generate and add noise
        noise = randn(size(solver.y_data_matrix)) * noise_std
        solver.y_data_matrix .+= noise
    end
    
    # Initialize dictionary
    solver.dictionary = randn(solver.dict_rows, solver.dict_cols)
    normalize_columns!(solver.dictionary)
    
    return solver
end

"""
Solve the DCDL optimization problem.
"""
function solve!(solver::DCDLSolver)
    for iter_num in 1:solver.max_iter
        # Sparse coding phase
        x_coef_matrix_former = copy(solver.x_coef_matrix)
        
        # Penalty MCP (Minimax Concave Penalty)
        x_coef_abs = abs.(x_coef_matrix_former)
        z = similar(x_coef_matrix_former)
        
        # Careful thresholding
        z .= ifelse.(
            x_coef_abs .> solver.λ * solver.γ, 
            solver.λ .* sign.(x_coef_matrix_former), 
            x_coef_matrix_former ./ solver.γ
        )
        
        # Gradient computation
        dt = solver.dictionary'
        dsq = dt * solver.dictionary
        phi = norm(dsq)
        
        # Preallocate gradient matrix
        grad = similar(solver.x_coef_matrix)
        for ii in 1:size(solver.y_data_matrix, 2)
            grad[:, ii] = dt * (solver.dictionary * x_coef_matrix_former[:, ii] - solver.y_data_matrix[:, ii])
        end
        
        # Coefficient update with careful broadcasting
        u_update = x_coef_matrix_former - grad ./ phi
        solver.x_coef_matrix .= sign.(u_update) .* max.(
            abs.(u_update .+ (1/phi) .* z) .- solver.λ/phi, 
            0.0
        )
        
        # Dictionary updating phase
        A = solver.x_coef_matrix * solver.x_coef_matrix'
        B = solver.y_data_matrix * solver.x_coef_matrix'
        
        # Compute omega safely
        omega = vec(sum(abs.(A), dims=1))
        omega = max.(omega, solver.epsilon)
        
        # Inner iteration for dictionary update
        for _ in 1:solver.inner_iter_max
            dictionary_former = copy(solver.dictionary)
            
            # Careful matrix multiplication and broadcasting
            dictionary_hat = solver.dictionary .* reshape(omega, 1, :) - (solver.dictionary * A - B)
            
            # Update each column carefully
            for j in 1:size(dictionary_hat, 2)
                solver.dictionary[:, j] .= dictionary_hat[:, j] ./ max(omega[j], norm(dictionary_hat[:, j]))
            end
            
            # Normalize dictionary
            normalize_columns!(solver.dictionary)
            
            # Check convergence
            if norm(dictionary_former - solver.dictionary) < solver.tolerance
                break
            end
        end
        
        # Evaluation
        solver.recovery_ratio[iter_num] = dictionaries_distance(
            solver.dictionary_true, 
            solver.dictionary
        )
        solver.total_error[iter_num] = sqrt(
            sum((solver.y_data_matrix - solver.dictionary * solver.x_coef_matrix).^2) / 
            length(solver.y_data_matrix)
        )
        
        println(
            "iter=$iter_num / $(solver.max_iter)  ",
            "totalErr=$(solver.total_error[iter_num])  ",
            "recoveryRatio=$(solver.recovery_ratio[iter_num])"
        )
    end
    
    return solver
end

"""
Plot the results of the DCDL algorithm.

Args:
- solver: DCDLSolver instance
- save_path: Optional path to save the plot
"""
function plot_results(
    solver::DCDLSolver, 
    save_path::Union{String, Nothing} = nothing
)
    p = plot(layout = grid(2, 1), 
        size = (800, 600), 
        link = :x
    )
    
    # Recovery Ratio subplot
    plot!(p[1, 1], 
        solver.recovery_ratio, 
        title = "Recovery Ratio", 
        xlabel = "Iteration", 
        ylabel = "Recovery Ratio",
        ylims = (0, 100),
        label = false
    )
    
    # Total Error subplot
    plot!(p[2, 1], 
        solver.total_error, 
        title = "Total Error", 
        xlabel = "Iteration", 
        ylabel = "Error",
        label = false
    )
    
    # Save or display plot
    if save_path !== nothing
        savefig(p, save_path)
    else
        display(p)
    end
    
    return p
end

"""
Manual command-line argument parsing function.

Args:
- args: Command-line arguments array

Returns:
- Dictionary of parsed arguments
"""
function parse_cli_args(args::Vector{String} = ARGS)
    # Default parameters
    params = Dict{String, Any}(
        "lambda" => 0.1,
        "gamma" => 50000.0,
        "max_iter" => 1000,
        "snr" => 20.0,
        "dict_rows" => 30,
        "dict_cols" => 50,
        "cardinality" => 1,
        "plot_save" => nothing
    )
    
    # Parse arguments
    i = 1
    while i <= length(args)
        # Check for key-value pairs
        if startswith(args[i], "--")
            key = replace(args[i], "--" => "")
            
            # Handle case where no value is provided
            if i + 1 > length(args) || startswith(args[i+1], "--")
                error("No value provided for argument $(args[i])")
            end
            
            # Convert to appropriate type
            value = args[i+1]
            params[key] = if key in ["max_iter", "dict_rows", "dict_cols", "cardinality"]
                parse(Int, value)
            elseif key in ["lambda", "gamma", "snr"]
                parse(Float64, value)
            else
                value
            end
            
            i += 2
        else
            i += 1
        end
    end
    
    return params
end

"""
Main function to run the DCDL algorithm.
"""
function main()
    # Parse command-line arguments
    try
        args = parse_cli_args()
        
        # Create and solve DCDL problem
        solver = DCDLSolver(
            λ = args["lambda"],
            γ = args["gamma"],
            max_iter = args["max_iter"],
            snr = args["snr"],
            dict_rows = args["dict_rows"],
            dict_cols = args["dict_cols"],
            cardinality = args["cardinality"]
        )
        
        # Generate data and solve
        generate_data!(solver)
        solve!(solver)
        
        # Plot results
        plot_results(solver, args["plot_save"])
    catch e
        println("Error parsing arguments: ", e)
        println("Usage: julia --project -e 'using DCDL; DCDL.main()' " *
                "--lambda VALUE --gamma VALUE --max_iter VALUE " *
                "--snr VALUE --dict_rows VALUE --dict_cols VALUE " *
                "--cardinality VALUE --plot_save PATH")
    end
end


# Export key functions and types
export DCDLSolver, 
       generate_data!, 
       solve!, 
       plot_results, 
       dictionaries_distance,
       main

end # module
