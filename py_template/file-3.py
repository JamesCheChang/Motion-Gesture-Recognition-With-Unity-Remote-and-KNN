import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def collect_all_data():
    # Load Test Data
    try:
        data_test = scipy.io.loadmat('rawPhoneSensorData_test.mat', simplify_cells=True)
        # Load Train Data
        data_train = scipy.io.loadmat('rawPhoneSensorData_train.mat', simplify_cells=True)
    except FileNotFoundError:
        print("Error: .mat files not found. Please ensure files are in the directory.")
        return None, None, None, None

    # Merge Data
    # Note: MATLAB [A; B] concatenates vertically (axis 0)
    # MATLAB [A B] concatenates horizontally (axis 1) for 1D arrays
    
    total_acc_x = np.vstack((data_train['total_acc_x_train'], data_test['total_acc_x_train']))
    total_acc_y = np.vstack((data_train['total_acc_y_train'], data_test['total_acc_y_train']))
    total_acc_z = np.vstack((data_train['total_acc_z_train'], data_test['total_acc_z_train']))
    
    # Merge Labels (assuming they are arrays of strings/chars)
    # MATLAB concatenates 1D arrays horizontally
    train_activity = np.concatenate((data_train['trainActivity'], data_test['trainActivity']))

    # --- Visualization Logic (Optional: Mimics the plotting in original script) ---
    # To enable plotting, set plot_on = True
    plot_on = False 
    
    if plot_on:
        # Helper to plot
        def plot_sample(k, title_prefix=""):
            fig, axs = plt.subplots(3, 1, figsize=(6, 8))
            axs[0].plot(total_acc_x[k, :])
            axs[0].set_ylabel('X-Value')
            axs[1].plot(total_acc_y[k, :])
            axs[1].set_ylabel('Y-Value')
            axs[2].plot(total_acc_z[k, :])
            axs[2].set_ylabel('Z-Value')
            axs[2].set_xlabel('Frame Index')
            if title_prefix:
                axs[0].set_title(f"{title_prefix} - Index {k}")
            plt.tight_layout()
            plt.show()

        # Iterate and categorize (Pythonic way using dictionaries instead of structs)
        activities = {'jump': [], 'rotate': [], 'wheel': []}
        
        for k in range(len(total_acc_z)):
            act = str(train_activity[k]).strip() # Handle numpy string/unicode
            if act in activities:
                 # Just storing indices or data for checking
                 pass 
                 # plot_sample(k, act) # Uncomment to plot every sample

    return total_acc_x, total_acc_y, total_acc_z, train_activity
