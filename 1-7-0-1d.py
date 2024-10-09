import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def generate_1d_interval(num_points):
    return np.random.uniform(0, 1, num_points)

def correlation_sum(points, epsilon):
    n = len(points)
    distances = np.abs(points[:, np.newaxis] - points)
    return np.sum(distances < epsilon) / (n * (n - 1))

def calculate_correlation_dimension(points, epsilon_range):
    c_epsilons = [correlation_sum(points, eps) for eps in epsilon_range]
    log_c = np.log(c_epsilons)
    log_eps = np.log(epsilon_range)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_eps, log_c)
    return slope, r_value**2

def main():
    num_points = 10000
    points = generate_1d_interval(num_points)
    
    epsilon_range = np.logspace(-3, -1, 50)
    
    correlation_dim, r_squared = calculate_correlation_dimension(points, epsilon_range)
    
    print(f"Estimated correlation dimension: {correlation_dim:.4f}")
    print(f"R-squared value: {r_squared:.4f}")
    
    c_epsilons = [correlation_sum(points, eps) for eps in epsilon_range]
    plt.loglog(epsilon_range, c_epsilons, 'bo-')
    plt.xlabel('ε')
    plt.ylabel('C(ε)')
    plt.title('Correlation of a 1D Interval')
    plt.grid(True)

    log_eps = np.log(epsilon_range)
    log_c = np.log(c_epsilons)
    fit = np.polyfit(log_eps, log_c, 1)
    fit_fn = np.poly1d(fit)
    plt.loglog(epsilon_range, np.exp(fit_fn(log_eps)), 'r-', label='Fit')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
