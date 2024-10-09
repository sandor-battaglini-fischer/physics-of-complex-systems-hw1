import numpy as np
from scipy.integrate import odeint
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def lorenz_system(state, t, sigma, rho, beta):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def generate_lorenz_attractor(sigma, rho, beta, initial_state, num_points, dt):
    t = np.linspace(0, num_points * dt, num_points)
    trajectory = odeint(lorenz_system, initial_state, t, args=(sigma, rho, beta))
    return trajectory[int(num_points/2):]  

def correlation_integral(points, r_range):
    nbrs = NearestNeighbors(n_neighbors=len(points), metric='euclidean').fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    c_r = []
    for r in r_range:
        count = np.sum(distances < r) - len(points)
        c_r.append(count / (len(points) * (len(points) - 1)))
    
    return np.array(c_r)

def estimate_correlation_dimension(points, r_range, fit_start=None, fit_end=None):
    c_r = correlation_integral(points, r_range)
    log_c_r = np.log(c_r)
    log_r = np.log(r_range)
    
    # Calculate local slopes
    local_slopes = np.diff(log_c_r) / np.diff(log_r)
    
    if fit_start is None or fit_end is None:
        # Automatic method to find scaling region
        window_size = 20
        smooth_slopes = np.convolve(local_slopes, np.ones(window_size)/window_size, mode='valid')
        scaling_start = np.argmin(np.abs(r_range[:-1] - 0.25))
        scaling_end = np.argmin(np.abs(r_range[:-1] - 0.8))
    else:
        scaling_start = np.argmin(np.abs(r_range - fit_start))
        scaling_end = np.argmin(np.abs(r_range - fit_end))
    
    # Linear fit in the scaling region
    coeffs = np.polyfit(log_r[scaling_start:scaling_end], log_c_r[scaling_start:scaling_end], 1)
    return coeffs[0], scaling_start, scaling_end, local_slopes


def automatic_scaling_region(log_r, log_c_r):
    # Algorithm from Ji, CuiCui and Zhu, Hua and Jiang, Wei
    # Step 1: Compute slopes
    slopes = np.diff(log_c_r) / np.diff(log_r)
    
    finite_mask = np.isfinite(slopes)
    slopes_clean = slopes[finite_mask]
    log_r_clean = log_r[:-1][finite_mask]
    
    if len(slopes_clean) == 0:
        raise ValueError("No valid slope values available after cleaning NaNs and infinities.")
    
    # Combine the cleaned log_r and slopes
    X = np.column_stack([log_r_clean, slopes_clean])
    
    # Step 2: Apply KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    labels = kmeans.labels_
    
    # Separate clusters and identify the one with least fluctuation (scaling region)
    cluster_0_std = np.std(slopes_clean[labels == 0])
    cluster_1_std = np.std(slopes_clean[labels == 1])
    scaling_region_label = 0 if cluster_0_std < cluster_1_std else 1
    
    # Retain points from the most stable cluster
    scaling_region = X[labels == scaling_region_label]
    
    # Step 3: Refine with point-slope-error algorithm
    sm = np.mean(scaling_region[:, 1])
    sigma = np.std(scaling_region[:, 1])
    error_range = (sm - 2 * sigma, sm + 2 * sigma)
    
    final_scaling_region = scaling_region[(scaling_region[:, 1] >= error_range[0]) & 
                                          (scaling_region[:, 1] <= error_range[1])]

    scaling_start_idx = np.argmin(np.abs(log_r - final_scaling_region[0, 0]))
    scaling_end_idx = np.argmin(np.abs(log_r - final_scaling_region[-1, 0]))
    
    return scaling_start_idx, scaling_end_idx
















def main():
    sigma, rho, beta = 10, 28, 8/3
    initial_state = [1, 1, 1]
    num_points = 10000
    dt = 0.001

    trajectory = generate_lorenz_attractor(sigma, rho, beta, initial_state, num_points, dt)
    
    r_range = np.logspace(-1.9, 1, 200)

    # Manual scaling region fit
    fit_start = 0.25  
    fit_end = 0.8
    
    correlation_dim_manual, scaling_start_manual, scaling_end_manual, local_slopes_manual = estimate_correlation_dimension(
        trajectory, r_range, fit_start=fit_start, fit_end=fit_end
    )
    
    # Automatic scaling region fit
    log_c_r = np.log(correlation_integral(trajectory, r_range))
    log_r = np.log(r_range)
    scaling_start_auto, scaling_end_auto = automatic_scaling_region(log_r, log_c_r)
    
    coeffs_auto = np.polyfit(log_r[scaling_start_auto:scaling_end_auto], 
                             log_c_r[scaling_start_auto:scaling_end_auto], 1)
    correlation_dim_auto = coeffs_auto[0]
    
    
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    c_r = correlation_integral(trajectory, r_range)
    ax1.loglog(r_range, c_r, '#1f77b4', label='Correlation Integral') 
    ax1.loglog(r_range[scaling_start_manual:scaling_end_manual], 
               np.exp(correlation_dim_manual * np.log(r_range[scaling_start_manual:scaling_end_manual]) + 
                      np.log(c_r[scaling_start_manual]) - correlation_dim_manual * np.log(r_range[scaling_start_manual])), 
               'r--', label='Manual Fit (Scaling Region)')
    ax1.loglog(r_range[scaling_start_auto:scaling_end_auto], 
               np.exp(correlation_dim_auto * np.log(r_range[scaling_start_auto:scaling_end_auto]) + 
                      np.log(c_r[scaling_start_auto]) - correlation_dim_auto * np.log(r_range[scaling_start_auto])), 
               'g--', label='Auto Fit (Scaling Region)')
    
    ax1.set_ylabel('Correlation Integral')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    ax2.semilogx(r_range[:-1], local_slopes_manual, 'k-', label='Local Slope')
    ax2.axhline(y=correlation_dim_manual, color='r', linestyle='--', label='Estimated Dimension (Manual)')
    ax2.axhline(y=correlation_dim_auto, color='g', linestyle='--', label='Estimated Dimension (Auto)')
    ax2.axvline(x=r_range[scaling_start_manual], color='k', linestyle=':', label='Manual Fit Range')
    ax2.axvline(x=r_range[scaling_end_manual], color='k', linestyle=':')
    ax2.axvline(x=r_range[scaling_start_auto], color='grey', linestyle='--', label='Auto Fit Range')
    ax2.axvline(x=r_range[scaling_end_auto], color='grey', linestyle='--')
    
    ax2.set_xlabel('Neighborhood Radius')
    ax2.set_ylabel('Local Slope')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.suptitle(f'Manual Correlation Dimension: {correlation_dim_manual:.5f}, Auto Correlation Dimension: {correlation_dim_auto:.5f}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()