import numpy as np
import pandas as pd
from utils import w_mean, w_std, test_pca

def get_data(Xraw, L, extra_m=None):
    """
    Feature extraction: Mean, Std, PCA
    Xraw: (N, 3*L) concatenated inputs
    """
    N = Xraw.shape[0]
    
    # Split back into X, Y, Z components
    acc_x = Xraw[:, 0:L]
    acc_y = Xraw[:, L:2*L]
    acc_z = Xraw[:, 2*L:3*L]
    
    # 1. Mean and Std
    # Stack features horizontally
    f_mean_x = w_mean(acc_x)
    f_mean_y = w_mean(acc_y)
    f_mean_z = w_mean(acc_z)
    
    f_std_x = w_std(acc_x)
    f_std_y = w_std(acc_y)
    f_std_z = w_std(acc_z)
    
    # 2. PCA
    # Pre-allocate arrays for PCA features
    pca1_x, pca1_y, pca1_z = [], [], []
    
    # We need to compute PCA parameters (mu, V) on the whole batch 
    # OR per sample if that was the intent. 
    # The MATLAB script calls test_PCA(total_acc_x_train', ...)
    # which implies passing the WHOLE dataset transposed (L x N).
    
    # Transpose for PCA function: (L, N) -> rows are time, cols are observations?
    # Actually, MATLAB script: test_PCA(total_acc_x_train', ...)
    # Inside test_PCA: mu = mean(X, 2). If X is (L, N), mean is across N (columns).
    # This seems to calculate ONE global PCA basis for the whole dataset?
    # Let's follow the MATLAB logic strictly.
    
    def process_component(data_block, extra_params):
        # data_block is (N, L). Transpose to (L, N) for the util function
        X_t = data_block.T 
        
        mu_in = extra_params['mu'] if extra_params else None
        V_in = extra_params['V'] if extra_params else None
        
        # Returns PCA1 signal (1xN), mu, V, PCA2
        p1, mu_out, V_out, _ = test_pca(X_t, mu_in, V_in)
        
        # In MATLAB code: T_pca table column seems to be the projected signal?
        # Actually, MATLAB `varfun` usually aggregates. 
        # But the custom logic returns `Wpca1_total_acc_x_train`.
        # For simplicity, let's assume we use the first component projection as a feature?
        # WARNING: The MATLAB script returns a TIME SERIES (1xN) per component?
        # No, `test_PCA` returns `PCA1_signal = v1.' * Xc`. If Xc is (3, N), signal is (1, N).
        # Wait, the MATLAB `test_PCA` input is `total_acc_x_train'`. 
        # If input is (L, N), `mu` is (L, 1). `v1` is (L, 1). 
        # `PCA1_signal` is (1, N). 
        # So for every observation (column), we get 1 value? 
        # Yes, it effectively projects the L-length waveform onto the principal time-basis.
        
        return p1, mu_out, V_out

    # Handle PCA Parameters (Train vs Test logic)
    extra_x = {'mu': extra_m['mu_x'], 'V': extra_m['V_x']} if extra_m else None
    extra_y = {'mu': extra_m['mu_y'], 'V': extra_m['V_y']} if extra_m else None
    extra_z = {'mu': extra_m['mu_z'], 'V': extra_m['V_z']} if extra_m else None
    
    p1_x, mu_x, V_x = process_component(acc_x, extra_x)
    p1_y, mu_y, V_y = process_component(acc_y, extra_y)
    p1_z, mu_z, V_z = process_component(acc_z, extra_z)

    # Combine all features into matrix X_out (N x D)
    # Features: Mean(3), Std(3), PCA1(3)
    X_out = np.column_stack((
        f_mean_x, f_mean_y, f_mean_z,
        f_std_x,  f_std_y,  f_std_z,
        p1_x,     p1_y,     p1_z
    ))

    # Store extras for training phase
    new_extras = {
        'mu_x': mu_x, 'V_x': V_x,
        'mu_y': mu_y, 'V_y': V_y,
        'mu_z': mu_z, 'V_z': V_z
    }
    
    return X_out, new_extras

def train_custom_knn(Xraw, Y, L):
    """Train 1-NN Model"""
    # Feature Extraction
    X, extra = get_data(Xraw, L, extra_m=None)
    
    # Standardization
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1 # Avoid div by zero
    
    Xstd = (X - mu) / sigma
    
    model = {
        'XTrain': Xstd,
        'YTrain': Y,
        'Mu': mu,
        'Sigma': sigma,
        'extra': extra
    }
    return model

def predict_custom_knn(model, XnewRaw, L):
    """Predict using 1-NN Model"""
    Xnew, _ = get_data(XnewRaw, L, extra_m=model['extra'])
    
    # Standardize
    Xstd = (Xnew - model['Mu']) / model['Sigma']
    Xstd = np.nan_to_num(Xstd) # Handle NaNs
    
    Xtrain = model['XTrain']
    Ytrain = model['YTrain']
    
    M = Xstd.shape[0]
    Yhat = np.empty(M, dtype=Ytrain.dtype)
    
    # 1-NN Euclidean Distance
    # This can be optimized using scipy.spatial.distance.cdist, 
    # but implemented loop-wise to match MATLAB logic.
    for i in range(M):
        xi = Xstd[i, :]
        # Sum of squared differences
        d2 = np.sum((Xtrain - xi)**2, axis=1)
        idx_min = np.argmin(d2)
        Yhat[i] = Ytrain[idx_min]
        
    return Yhat

def train_classifier_cv(total_acc_x, total_acc_y, total_acc_z, labels):
    """Driver for Cross-Validation"""
    # Combine inputs
    X = np.hstack((total_acc_x, total_acc_y, total_acc_z))
    L = 41 # Waveform length
    K = 7  # Folds
    N = X.shape[0]
    
    # Deterministic folds: 0, 1, ..., 6, 0, 1 ...
    fold_index = np.arange(N) % K
    
    validation_predictions = np.empty_like(labels)
    
    print(f"Starting {K}-fold Cross Validation on {N} samples...")
    
    for k in range(K):
        test_mask = (fold_index == k)
        train_mask = ~test_mask
        
        X_train = X[train_mask, :]
        Y_train = labels[train_mask]
        X_test = X[test_mask, :]
        
        # Train
        model = train_custom_knn(X_train, Y_train, L)
        
        # Predict
        Y_hat = predict_custom_knn(model, X_test, L)
        
        validation_predictions[test_mask] = Y_hat
        
    # Accuracy
    accuracy = np.mean(validation_predictions == labels)
    return validation_predictions, accuracy
