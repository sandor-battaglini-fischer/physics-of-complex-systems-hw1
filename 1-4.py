import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


sigma = 10
b = 8 / 3
rho = 28

def lorenz_system(state, t):
    x, y, z = state 
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - b * z
    return [dx_dt, dy_dt, dz_dt]

def runge_kutta_lorenz(x0, y0, z0, t0, tf, dt, sigma, rho, beta):
    N = int((tf - t0) / dt)
    t = np.zeros(N + 1)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    z = np.zeros(N + 1)
    t[0], x[0], y[0], z[0] = t0, x0, y0, z0
    for i in range(N):
        xi, yi, zi = x[i], y[i], z[i]

        k1x = sigma * (yi - xi)
        k1y = xi * (rho - zi) - yi
        k1z = xi * yi - beta * zi

        xk2 = xi + 0.5 * k1x * dt
        yk2 = yi + 0.5 * k1y * dt
        zk2 = zi + 0.5 * k1z * dt
        k2x = sigma * (yk2 - xk2)
        k2y = xk2 * (rho - zk2) - yk2
        k2z = xk2 * yk2 - beta * zk2

        xk3 = xi + 0.5 * k2x * dt
        yk3 = yi + 0.5 * k2y * dt
        zk3 = zi + 0.5 * k2z * dt
        k3x = sigma * (yk3 - xk3)
        k3y = xk3 * (rho - zk3) - yk3
        k3z = xk3 * yk3 - beta * zk3

        xk4 = xi + k3x * dt
        yk4 = yi + k3y * dt
        zk4 = zi + k3z * dt
        k4x = sigma * (yk4 - xk4)
        k4y = xk4 * (rho - zk4) - yk4
        k4z = xk4 * yk4 - beta * zk4

        x[i + 1] = xi + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y[i + 1] = yi + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        z[i + 1] = zi + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        t[i + 1] = t[i] + dt
    return t, x, y, z

def integrate_lorenz(initial_state, t):
    # return odeint(lorenz_system, initial_state, t)
    x0, y0, z0 = initial_state
    t0, tf = t[0], t[-1]
    dt = t[1] - t[0]
    t, x, y, z = runge_kutta_lorenz(x0, y0, z0, t0, tf, dt, sigma, rho, b)
    return np.column_stack((x, y, z))

def calculate_lyapunov_exponent(initial_state, perturbation, t_max, dt):
    t = np.arange(0, t_max, dt)
    
    trajectory = integrate_lorenz(initial_state, t)
    
    perturbed_state = initial_state.copy() 
    perturbed_state += perturbation
    perturbed_trajectory = integrate_lorenz(perturbed_state, t)
    
    divergence = np.linalg.norm(trajectory - perturbed_trajectory, axis=1)
    
    lyap = np.log(divergence / np.linalg.norm(perturbation))
    return t, lyap

initial_state = np.array([1.0, 1.0, 1.0])
perturbation = np.array([1e-10, 1e-10, 1e-10])  
t_max = 100
dt = 0.01

t, lyap = calculate_lyapunov_exponent(initial_state, perturbation, t_max, dt)

plt.figure(figsize=(10, 6))
plt.plot(t, lyap, label='Lyapunov Exponent')
plt.xlabel('Time')
plt.ylabel('Estimated Lyapunov Exponent')
# plt.title('Estimation of Maximum Lyapunov Exponent for Lorenz System')
plt.grid(True)
plt.legend()
plt.show()

average_lyap = np.mean(lyap[int(len(lyap)/2):])
print(f"Estimated maximum Lyapunov exponent: {average_lyap:.4f}")