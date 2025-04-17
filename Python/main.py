from arguments import parse_arguments
from solvers import DCDL


def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create and solve DCDL problem
    dcdl = DCDL(
        lambda_val=args.lambda_val,
        gamma=args.gamma,
        max_iter=args.max_iter,
        snr=args.snr,
        dict_rows=args.dict_rows,
        dict_cols=args.dict_cols,
        cardinality=args.cardinality
    )
    
    # Generate data and solve
    dcdl.generate_data()
    dcdl.solve()
    
    # Plot results
    dcdl.plot_results(save_path=args.plot_save)

if __name__ == "__main__":
    main()