import numpy as np

def w_mean(X):
    """Row-wise mean"""
    return np.mean(X, axis=1)

def w_std(X):
    """Row-wise std"""
    return np.std(X, axis=1, ddof=1) # ddof=1 matches MATLAB default

def test_pca(X, mu=None, V=None):
    """
    Performs PCA on signal X (3xN accelerometer window).
    X shape: (3, N)
    """
    # Ensure X is numpy array
    X = np.array(X)
    
    if mu is None and V is None:
        # Step 1: Mean removal
        mu = np.mean(X, axis=1, keepdims=True)
        Xc = X - mu
        
        # Step 2: Covariance matrix
        # MATLAB cov(Xc.') expects observations in rows. 
        # Xc.' is (N, 3).
        # np.cov expects variables in rows by default, so we can pass Xc directly (3, N)
        C = np.cov(Xc) 
        
        # Step 3: Eigen-decomposition
        # eigh is for symmetric matrices (covariance always is)
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Sort descending (eigh returns ascending)
        idx = np.argsort(eigvals)[::-1]
        V = eigvecs[:, idx]
        
    else:
        # Use provided stats
        Xc = X - mu
    
    # PCA1 direction (eigenvector corresponding to largest eigenvalue)
    v1 = V[:, 0] # Shape (3,)
    v2 = V[:, 1] # Shape (3,)
    
    # Project data
    # v1 is (3,), Xc is (3, N). Result (N,)
    pca1_signal = np.dot(v1.T, Xc)
    pca2_signal = np.dot(v2.T, Xc)
    
    return pca1_signal, mu, V, pca2_signal
