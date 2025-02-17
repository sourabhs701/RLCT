import numpy as np
from scipy.linalg import sqrtm


def compute_fid(real_data, synthetic_data):
    # Compute mean and covariance of real and synthetic data
    mu_real, sigma_real = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
    mu_synth, sigma_synth = np.mean(synthetic_data, axis=0), np.cov(synthetic_data, rowvar=False)

    # Compute squared difference of means
    mean_diff = np.sum((mu_real - mu_synth) ** 2)

    # Compute square root of product of covariance matrices
    cov_mean = sqrtm(sigma_real @ sigma_synth)

    # Handle numerical instability
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = mean_diff + np.trace(sigma_real + sigma_synth - 2 * cov_mean)
    return fid 