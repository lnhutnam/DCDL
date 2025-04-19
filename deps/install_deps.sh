#!/bin/bash

# Check if Julia is installed
if ! command -v julia &> /dev/null; then
    echo "❌ Julia is not installed. Please install Julia first."
    exit 1
fi

# Run the Julia dependency installation script
julia --project install_deps.jl