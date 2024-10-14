import numpy as np
from multiprocessing import Pool
import h5py


def make_positive_semidefinite(matrix):
    # Force symmetry
    matrix = (matrix + matrix.T) / 2

    # Perform eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(matrix)

    # Debug: Check eigenvalues before adjustment
    if np.any(np.isnan(eigvals)):
        print("NaNs in eigenvalues before adjustment.")

    # Set any negative or near-zero eigenvalues to a small positive number
    eigvals[eigvals < 0] = 1e-10
    eigvals[eigvals < 1e-10] += 1e-10

    # Debug: Check eigenvalues after adjustment
    if np.any(np.isnan(eigvals)):
        print("NaNs in eigenvalues after adjustment.")

    # Reconstruct the matrix
    matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Ensure the result is symmetric
    matrix = (matrix + matrix.T) / 2

    return matrix


def sample_block(args):
    mean_block, cov_block, num_samples = args

    # Ensure that cov_block is positive semidefinite
    cov_block = make_positive_semidefinite(cov_block)

    # Perform eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_block)

    # Debug: Check for NaNs in eigvals or eigvecs
    if np.any(np.isnan(eigvals)):
        print("NaNs detected in eigenvalues during sampling.")
    if np.any(np.isnan(eigvecs)):
        print("NaNs detected in eigenvectors during sampling.")

    # Sample from a standard normal distribution
    z = np.random.randn(len(mean_block), num_samples)

    # Transform the samples using the eigenvalues and eigenvectors
    samples = eigvecs @ np.diag(np.sqrt(eigvals)) @ z + mean_block[:, np.newaxis]

    # Debug: Check for NaNs in the generated samples
    if np.any(np.isnan(samples)):
        print("NaNs detected in the samples.")

    return samples.T if num_samples > 1 else samples.ravel()


def blockwise_sampling_parallel(mean, cov_matrix_path, block_size, num_samples, num_workers):
    n = len(mean)
    blocks = []

    # Prepare arguments for parallel processing
    tasks = []
    with h5py.File(cov_matrix_path, 'r') as f:
        cov_matrix = f['cov_matrix']
        for i in range(0, n, block_size):
            # Load only the relevant block from the covariance matrix
            cov_block = cov_matrix[i:i + block_size, i:i + block_size]
            mean_block = mean[i:i + block_size]
            tasks.append((mean_block, cov_block, num_samples))

    # Use Pool to parallelize the sampling
    with Pool(num_workers) as pool:
        block_samples = pool.map(sample_block, tasks)

    # Combine all the block samples to form the full sample
    full_samples = np.hstack(block_samples)

    return full_samples


# Example usage
mean = np.random.rand(65536)  # Assuming mean fits in memory

# Create the covariance matrix on disk
cov_matrix_path = 'checkpoint/cov_matrix.h5'
with h5py.File(cov_matrix_path, 'w') as f:
    cov_matrix = f.create_dataset('cov_matrix', (65536, 65536), dtype='float32')

    # Create a symmetric positive semidefinite covariance matrix
    for i in range(65536):
        row = np.random.randn(65536)  # Generate random row
        cov_matrix[i, :] = row @ row.T  # Create a symmetric positive semidefinite matrix

block_size = 100  # Adjust based on memory capacity
num_samples = 1  # Number of samples to generate
num_workers = 20  # Number of parallel workers (cores)

test_samples = blockwise_sampling_parallel(mean, cov_matrix_path, block_size, num_samples, num_workers)
