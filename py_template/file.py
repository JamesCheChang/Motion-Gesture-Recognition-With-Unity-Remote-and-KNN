import scipy.io
import numpy as np
import os

def convert_mat_to_npz(mat_file_path, npz_file_path):
    if not os.path.exists(mat_file_path):
        print(f"File not found: {mat_file_path}")
        return

    # Load mat file
    # simplify_cells=True helps convert MATLAB cells to numpy arrays automatically
    mat_data = scipy.io.loadmat(mat_file_path, simplify_cells=True)
    
    # Filter out internal MATLAB variables (start with __)
    data_dict = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    
    # Save as compressed numpy
    np.savez_compressed(npz_file_path, **data_dict)
    print(f"Converted {mat_file_path} -> {npz_file_path}")

# Usage for your files
# Assuming files are in current directory
convert_mat_to_npz('cor_data.mat', 'cor_data.npz')
convert_mat_to_npz('rawPhoneSensorData_test.mat', 'rawPhoneSensorData_test.npz')
convert_mat_to_npz('rawPhoneSensorData_train.mat', 'rawPhoneSensorData_train.npz')
