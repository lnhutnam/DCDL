import numpy as np
import matplotlib.pyplot as plt

from utils import dictionaries_distance

class DCDL:
    def __init__(
        self, 
        lambda_val=0.1, # penalty function: Minimax Concave Penalty (MCP), lambda > 0, gamma > 1
        gamma=50000, 
        max_iter=1000, 
        length_gain=30, 
        snr=20, # Noise. The modified parameters are possibly required when SNR is low.
        inner_iter_max=1, # Dictionary updating inneriiteration maximum. If 1 is hard to reach good result, try increasing, for instance, 10.
        tolerance=1e-5, # dictionary updating tolerance
        epsilon=1e-6, # numerical optimization 
        dict_rows=30, # For generating dictionary, row dimensional
        dict_cols=50, # For generating dictionary, col dimensional
        cardinality=1, # cardinality, the number of non-zeros
    ):
        """
        Initialize DCDL (Dictionary Learning with Difference of Convex programming) algorithm.
        
        Args:
            lambda_val (float): Penalty parameter
            gamma (float): Penalty function parameter
            max_iter (int): Maximum iterations
            length_gain (int): Length gain factor
            snr (float): Signal-to-noise ratio
            inner_iter_max (int): Maximum inner iterations for dictionary update
            tolerance (float): Convergence tolerance
            epsilon (float): Small numerical constant
            dict_rows (int): Number of rows in dictionary
            dict_cols (int): Number of columns in dictionary
            cardinality (int): Number of non-zero coefficients
        """
        # Algorithm parameters
        self.lambda_val = lambda_val
        self.gamma = gamma
        self.max_iter = max_iter
        self.length_gain = length_gain
        self.snr = snr
        self.inner_iter_max = inner_iter_max
        self.tolerance = tolerance
        self.epsilon = epsilon
        
        # Data generation parameters
        self.dict_rows = dict_rows
        self.dict_cols = dict_cols
        self.cardinality = cardinality
        
        # Initialization placeholders
        self.dictionary_true = None
        self.dictionary = None
        self.x_coef_matrix = None
        self.y_data_matrix = None
        
        # Results tracking
        self.recovery_ratio = np.zeros(max_iter)
        self.total_error = np.zeros(max_iter)
    
    def generate_data(self):
        """
        Generate ground truth dictionary and data based on initialized parameters.
        """
        # Generate ground truth dictionary
        self.dictionary_true = np.random.randn(self.dict_rows, self.dict_cols) # generate by using random 
        self.dictionary_true /= np.linalg.norm(self.dictionary_true, axis=0) # normalize ground truth dictionary
        
        # Calculate data length
        L = self.dict_cols * self.length_gain # dim_length, signal and coefficient
        
        # Generate coefficient matrix
        self.x_coef_matrix = np.zeros((self.dict_cols, L))
        self.x_coef_matrix[:self.cardinality, :] = np.random.randn(self.cardinality, L)
        
        # Randomize coefficient matrix
        for i in range(L):
            self.x_coef_matrix[:, i] = self.x_coef_matrix[np.random.permutation(self.dict_cols), i]
        
        # Generate data matrix
        self.y_data_matrix = self.dictionary_true @ self.x_coef_matrix
        
        # Add noise if SNR is finite
        if self.snr != np.inf:
            noise_std = np.std(self.y_data_matrix.ravel()) * 10**(-self.snr/20)
            noise = np.random.randn(*self.y_data_matrix.shape) * noise_std

            # Noise signal 
            self.y_data_matrix += noise
        
        # Initialize dictionary
        self.dictionary = np.random.randn(self.dict_rows, self.dict_cols)
        self.dictionary /= np.linalg.norm(self.dictionary, axis=0) # normalize generated dictionary
    
    def solve(self):
        """
        Main DCDL algorithm solving process
        """
        for iter_num in range(self.max_iter):
            # Sparse coding phase
            x_coef_matrix_former = self.x_coef_matrix.copy()
            
            # Penalty MCP (Minimax Concave Penalty)
            x_coef_abs = np.abs(x_coef_matrix_former)
            z = np.zeros_like(x_coef_matrix_former)
            
            # Thresholding
            ind_high = x_coef_abs > self.lambda_val * self.gamma
            ind_low = x_coef_abs <= self.lambda_val * self.gamma
            
            z[ind_high] = self.lambda_val * np.sign(x_coef_matrix_former[ind_high])
            z[ind_low] = x_coef_matrix_former[ind_low] / self.gamma
            
            # Gradient computation
            dt = self.dictionary.T
            dsq = dt @ self.dictionary
            phi = np.linalg.norm(dsq)
            
            grad = np.zeros_like(self.x_coef_matrix)
            for ii in range(self.y_data_matrix.shape[1]):
                grad[:, ii] = dt @ (self.dictionary @ x_coef_matrix_former[:, ii] - self.y_data_matrix[:, ii])
            
            # Coefficient update
            u_update = x_coef_matrix_former - grad / phi
            self.x_coef_matrix = np.sign(u_update) * np.maximum(
                np.abs(u_update + (1/phi) * z) - self.lambda_val/phi, 
                0
            )
            
            # Dictionary updating phase
            A = self.x_coef_matrix @ self.x_coef_matrix.T
            B = self.y_data_matrix @ self.x_coef_matrix.T
            
            omega = np.abs(A) @ np.ones(A.shape[0])
            omega = np.maximum(omega, self.epsilon)
            
            omega_matrix = np.tile(omega, (self.dictionary.shape[0], 1))
            
            # Inner iteration for dictionary update
            for _ in range(self.inner_iter_max):
                dictionary_former = self.dictionary.copy()
                dictionary_hat = omega_matrix * self.dictionary - (self.dictionary @ A - B)
                
                for j in range(dictionary_hat.shape[1]):
                    self.dictionary[:, j] = dictionary_hat[:, j] / max(omega[j], np.linalg.norm(dictionary_hat[:, j]))
                
                # Normalize dictionary
                self.dictionary /= np.linalg.norm(self.dictionary, axis=0)
                
                # Check convergence
                if np.linalg.norm(dictionary_former - self.dictionary) < self.tolerance:
                    break
            
            # Evaluation
            self.recovery_ratio[iter_num] = dictionaries_distance(self.dictionary_true, self.dictionary)
            self.total_error[iter_num] = np.sqrt(
                np.sum((self.y_data_matrix - self.dictionary @ self.x_coef_matrix)**2) / 
                self.y_data_matrix.size
            )
            
            print(f'iter={iter_num+1} / {self.max_iter}  totalErr={self.total_error[iter_num]:.6f} recoveryRatio={self.recovery_ratio[iter_num]:.6f}')
        
        return self
    
    def plot_results(self, save_path=None):
        """
        Plot recovery ratio and total error
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.recovery_ratio)
        plt.title('Recovery Ratio')
        plt.xlabel('Iteration')
        plt.ylabel('Recovery Ratio')
        plt.ylim(0, 100)
        
        plt.subplot(2, 1, 2)
        plt.plot(self.total_error)
        plt.title('Total Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
