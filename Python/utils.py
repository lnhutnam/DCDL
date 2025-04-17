import numpy as np

def dictionaries_distance(original, new, threshold=0.01):
    """
    Calculate the recovery ratio between two dictionaries.
    
    Args:
        original (np.ndarray): Original dictionary
        new (np.ndarray): New dictionary
        threshold (float): Distance threshold
    
    Returns:
        float: Recovery ratio percentage
    """
    distances = np.abs(original.T @ new)
    counter = 0
    
    for i in range(original.shape[1]):
        min_value = 1 - np.max(distances[i, :])
        counter += int(min_value < threshold)
    
    return 100 * counter / original.shape[1]