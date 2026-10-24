import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def cov1(X):
    # MATLAB: X=X'; mean(X,2); C = cov(Xc.')
    # This implies input X is (Observations x Length).
    # We want covariance of the TIME points (Length x Length matrix)
    # Python cov expects (Vars, Obs). 
    # If X is (Obs, Length), X.T is (Length, Obs). 
    # np.cov(X.T) gives (Length, Length).
    X_t = X.T
    mu = np.mean(X_t, axis=1, keepdims=True)
    Xc = X_t - mu
    # np.cov defaults to rowvar=True (rows are variables)
    C = np.cov(Xc) 
    return C

def check_cov_matrix():
    try:
        # Load data (assuming converted to .npz or use .mat)
        d = scipy.io.loadmat('cor_data.mat', simplify_cells=True)
        # Assuming struct j_m, r_m, w_m exists in mat file
        j_m = d['j_m']
        r_m = d['r_m']
        w_m = d['w_m']
    except Exception as e:
        print(f"Could not load cor_data: {e}")
        return

    # Compute Covariances
    mats = {}
    for prefix, data in [('j_m', j_m), ('r_m', r_m), ('w_m', w_m)]:
        mats[f'co_{prefix}_x'] = cov1(data['x'])
        mats[f'co_{prefix}_y'] = cov1(data['y'])
        mats[f'co_{prefix}_z'] = cov1(data['z'])

    # Calculate global min/max for color scaling
    all_values = np.array([m for m in mats.values()])
    g_min = np.min(all_values)
    g_max = np.max(all_values)

    # Plotting Helper
    def show_matrix(key):
        plt.figure()
        plt.imshow(mats[key], aspect='auto', origin='lower', vmin=g_min, vmax=g_max)
        plt.colorbar()
        plt.title(key)
        plt.axis('tight')
        plt.show()

    # Visualize specific ones as requested
    keys_to_plot = [
        'co_j_m_x', 'co_j_m_y', 'co_j_m_z',
        'co_r_m_x', 'co_r_m_y', 'co_r_m_z',
        'co_w_m_x', 'co_w_m_y', 'co_w_m_z'
    ]
    
    for k in keys_to_plot:
        show_matrix(k)

    # Combined Subplot
    fig, ax = plt.subplots(3, 1, figsize=(6, 10))
    
    im1 = ax[0].imshow(mats['co_j_m_z'], origin='lower', aspect='auto', vmin=g_min, vmax=g_max)
    ax[0].set_title('Jump', fontsize=18)
    plt.colorbar(im1, ax=ax[0])
    
    im2 = ax[1].imshow(mats['co_r_m_z'], origin='lower', aspect='auto', vmin=g_min, vmax=g_max)
    ax[1].set_title('Rotate', fontsize=18)
    plt.colorbar(im2, ax=ax[1])
    
    im3 = ax[2].imshow(mats['co_w_m_z'], origin='lower', aspect='auto', vmin=g_min, vmax=g_max)
    ax[2].set_title('Wheel', fontsize=18)
    plt.colorbar(im3, ax=ax[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_cov_matrix()
