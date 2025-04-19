#!/usr/bin/env julia

"""
Dependency installation script for DCDL.jl project.

This script:
1. Checks for and installs required Julia packages
2. Ensures project environment is set up
3. Provides helpful output about installation process
"""

import Pkg

function install_dependencies()
    println("🔍 Checking and installing project dependencies...")
    
    # List of required packages
    required_packages = [
        "LinearAlgebra",
        "Plots",
        "Random", 
        "Statistics",
        "ArgParse"
    ]
    
    # Track installation status
    successful_installs = String[]
    failed_installs = String[]
    
    # Activate project environment
    try
        Pkg.activate(".")
        println("✅ Activated project environment")
    catch e
        println("❌ Error activating project environment: ", e)
        return false
    end
    
    # Install packages
    for pkg in required_packages
        try
            Pkg.add(pkg)
            push!(successful_installs, pkg)
            println("📦 Installed: ", pkg)
        catch e
            push!(failed_installs, pkg)
            println("❌ Failed to install: ", pkg)
            println("   Error: ", e)
        end
    end
    
    # Instantiate the project to ensure all dependencies are installed
    try
        Pkg.instantiate()
        println("🔗 Project dependencies synchronized")
    catch e
        println("❌ Error synchronizing dependencies: ", e)
    end
    
    # Print summary
    println("\n📊 Installation Summary:")
    println("Successful Installs: ", join(successful_installs, ", "))
    
    if !isempty(failed_installs)
        println("Failed Installs: ", join(failed_installs, ", "))
        return false
    end
    
    return true
end

function main()
    println("🚀 DCDL.jl Dependency Installer")
    println("================================")
    
    success = install_dependencies()
    
    if success
        println("\n✨ Dependencies installed successfully!")
        println("You can now use the DCDL.jl package.")
    else
        println("\n⚠️ Some dependencies failed to install.")
        println("Please check your Julia installation and network connection.")
    end
end

# Run the main function
main()