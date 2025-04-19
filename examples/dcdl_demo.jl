#!/usr/bin/env julia

"""
Comprehensive DCDL (Dictionary Learning with Difference of Convex) Project Demo

This script showcases various functionalities of the DCDL package:
1. Basic dictionary learning
2. Parameter tuning
3. Visualization
4. Performance analysis
"""

using DCDL
using Plots
using Random
using Statistics

"""
Generate synthetic dataset for dictionary learning.

Args:
- n_rows: Number of rows in dictionary
- n_cols: Number of columns in dictionary
- n_samples: Number of samples
- sparsity: Sparsity level of coefficients

Returns:
- Dictionary matrix
- Coefficient matrix
- Data matrix
"""
function generate_synthetic_data(
    n_rows::Int = 30, 
    n_cols::Int = 50, 
    n_samples::Int = 1000, 
    sparsity::Float64 = 0.1
)
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Generate true dictionary
    true_dict = randn(n_rows, n_cols)
    true_dict ./= norm.(eachcol(true_dict))
    
    # Generate sparse coefficients
    coeffs = zeros(n_cols, n_samples)
    for i in 1:n_samples
        # Create sparse coefficient vector
        active_indices = randperm(n_cols)[1:round(Int, n_cols * sparsity)]
        coeffs[active_indices, i] = randn(length(active_indices))
    end
    
    # Generate data matrix
    data_matrix = true_dict * coeffs
    
    return true_dict, coeffs, data_matrix
end

"""
Demonstrate dictionary learning with different parameters.

Args:
- data_matrix: Input data matrix
- title: Experiment title
"""
function demo_dictionary_learning(data_matrix, title::String = "DCDL Demo")
    println("\n🔍 Demonstration: $title")
    
    # Different parameter configurations
    parameter_sets = [
        (lambda = 0.1, gamma = 50000.0, max_iter = 1000, snr = 20.0),
        (lambda = 0.05, gamma = 60000.0, max_iter = 1500, snr = 25.0),
        (lambda = 0.2, gamma = 40000.0, max_iter = 800, snr = 15.0)
    ]
    
    # Store results for comparison
    results = []
    
    # Iterate through parameter configurations
    for (i, params) in enumerate(parameter_sets)
        println("\n📊 Parameter Set $i:")
        println("  λ (lambda): ", params.lambda)
        println("  γ (gamma): ", params.gamma)
        println("  Max Iterations: ", params.max_iter)
        println("  SNR: ", params.snr)
        
        # Create solver with current parameters
        solver = DCDLSolver(
            λ = params.lambda,
            γ = params.gamma,
            max_iter = params.max_iter,
            snr = params.snr,
            dict_rows = size(data_matrix, 1),
            dict_cols = 50,
            cardinality = 5
        )
        
        # Set data matrix
        solver.y_data_matrix = data_matrix
        
        # Solve the problem
        solve!(solver)
        
        # Store and analyze results
        push!(results, (
            params = params, 
            recovery_ratio = solver.recovery_ratio[end], 
            total_error = solver.total_error[end],
            dictionary = solver.dictionary
        ))
    end
    
    # Visualize results
    p = plot(layout = (2, 1), size = (800, 600))
    
    # Plot recovery ratio
    plot!(p[1], 
        [r.recovery_ratio for r in results], 
        title = "Recovery Ratio Comparison", 
        ylabel = "Recovery Ratio", 
        label = ["Set 1" "Set 2" "Set 3"]
    )
    
    # Plot total error
    plot!(p[2], 
        [r.total_error for r in results], 
        title = "Total Error Comparison", 
        ylabel = "Total Error", 
        label = ["Set 1" "Set 2" "Set 3"]
    )
    
    # Save plot
    savefig(p, "parameter_comparison.png")
    display(p)
    
    return results
end

"""
Main demonstration function.
"""
function main()
    println("🚀 DCDL Dictionary Learning Demonstration")
    println("=======================================")
    
    # Generate synthetic data
    true_dict, true_coeffs, data_matrix = generate_synthetic_data()
    
    # Demonstrate dictionary learning
    results = demo_dictionary_learning(data_matrix)
    
    # Print comparative analysis
    println("\n📈 Comparative Analysis:")
    for (i, result) in enumerate(results)
        println("\nParameter Set $i:")
        println("  Recovery Ratio: ", result.recovery_ratio)
        println("  Total Error: ", result.total_error)
    end
end

# Run the main demonstration
main()