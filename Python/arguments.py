import argparse

def parse_arguments():
    """
    Parse command-line arguments for DCDL algorithm.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Dictionary Learning with Difference of Convex Programming')
    
    # Algorithm parameters
    parser.add_argument('--lambda-val', type=float, default=0.1, 
                        help='Penalty parameter (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=50000, 
                        help='Penalty function parameter (default: 50000)')
    parser.add_argument('--max-iter', type=int, default=1000, 
                        help='Maximum iterations (default: 1000)')
    parser.add_argument('--snr', type=float, default=20, 
                        help='Signal-to-noise ratio (default: 20)')
    
    # Data generation parameters
    parser.add_argument('--dict-rows', type=int, default=30, 
                        help='Number of dictionary rows (default: 30)')
    parser.add_argument('--dict-cols', type=int, default=50, 
                        help='Number of dictionary columns (default: 50)')
    parser.add_argument('--cardinality', type=int, default=1, 
                        help='Number of non-zero coefficients (default: 1)')
    
    # Output parameters
    parser.add_argument('--plot-save', type=str, default=None, 
                        help='Path to save the results plot')
    
    return parser.parse_args()