import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
sigma = 10.0
rho = 400
beta = 8.0 / 3.0
dt = 0.001
t0 = 0.0
tf = 20.0
x0, y0, z0 = 1.0, 0.0, 0.0

# Euler Method
def euler_lorenz(x0, y0, z0, t0, tf, dt, sigma, rho, beta):
    N = int((tf - t0) / dt)
    t = np.zeros(N + 1)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    z = np.zeros(N + 1)
    t[0], x[0], y[0], z[0] = t0, x0, y0, z0
    for i in range(N):
        dx = sigma * (y[i] - x[i])
        dy = x[i] * (rho - z[i]) - y[i]
        dz = x[i] * y[i] - beta * z[i]
        x[i + 1] = x[i] + dx * dt
        y[i + 1] = y[i] + dy * dt
        z[i + 1] = z[i] + dz * dt
        t[i + 1] = t[i] + dt
    return t, x, y, z

# Runge-Kutta Method
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

# SciPy Integration
def lorenz(t, u, sigma, rho, beta):
    x, y, z = u
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Integration
t_euler, x_euler, y_euler, z_euler = euler_lorenz(
    x0, y0, z0, t0, tf, dt, sigma, rho, beta
)
t_rk, x_rk, y_rk, z_rk = runge_kutta_lorenz(
    x0, y0, z0, t0, tf, dt, sigma, rho, beta
)
t_span = (t0, tf)
t_eval = np.arange(t0, tf, dt)
u0 = [x0, y0, z0]
sol = solve_ivp(
    lorenz, t_span, u0, args=(sigma, rho, beta), t_eval=t_eval, method='RK45'
)
x_de = sol.y[0]
y_de = sol.y[1]
z_de = sol.y[2]





# Load reference data
ref_data = np.loadtxt("lorenz-ref.data")
t_ref = ref_data[:, 0]
u_ref = ref_data[:, 1:]

# Interpolate numerical solutions to match reference time points
from scipy.interpolate import interp1d

def interpolate_solution(t, x, y, z, t_ref):
    fx = interp1d(t, x, kind='linear', fill_value='extrapolate')
    fy = interp1d(t, y, kind='linear', fill_value='extrapolate')
    fz = interp1d(t, z, kind='linear', fill_value='extrapolate')
    return fx(t_ref), fy(t_ref), fz(t_ref)

x_euler_interp, y_euler_interp, z_euler_interp = interpolate_solution(t_euler, x_euler, y_euler, z_euler, t_ref)
x_rk_interp, y_rk_interp, z_rk_interp = interpolate_solution(t_rk, x_rk, y_rk, z_rk, t_ref)
x_de_interp, y_de_interp, z_de_interp = interpolate_solution(sol.t, x_de, y_de, z_de, t_ref)

# Calculate errors
def calculate_error(x, y, z, x_ref, y_ref, z_ref):
    return np.sqrt((x - x_ref)**2 + (y - y_ref)**2 + (z - z_ref)**2)

error_euler = calculate_error(x_euler_interp, y_euler_interp, z_euler_interp, u_ref[:, 0], u_ref[:, 1], u_ref[:, 2])
error_rk = calculate_error(x_rk_interp, y_rk_interp, z_rk_interp, u_ref[:, 0], u_ref[:, 1], u_ref[:, 2])
error_de = calculate_error(x_de_interp, y_de_interp, z_de_interp, u_ref[:, 0], u_ref[:, 1], u_ref[:, 2])

max_t = 200
t_mask = t_ref <= max_t

plt.figure(figsize=(10, 6))
plt.semilogy(t_ref[t_mask], error_euler[t_mask], label='Euler Method')
plt.semilogy(t_ref[t_mask], error_rk[t_mask], label='Runge-Kutta Method')
# plt.semilogy(t_ref[t_mask], error_de[t_mask], label='SciPy Integration (RK45)')
plt.xlabel('Time')
plt.ylabel('Error (Euclidean distance)')
plt.title(f'Error Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Print maximum errors up to t=200
print(f"Maximum error (Euler)")
print(f"Maximum error (Runge-Kutta)")
print(f"Maximum error (SciPy RK45)")






# Trajectories
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def calculate_fixed_points(sigma, rho, beta):
    C = np.sqrt(beta * (rho - 1))
    return [
        (0, 0, 0),  
        (C, C, rho - 1),  
        (-C, -C, rho - 1) 
    ]

fixed_points = calculate_fixed_points(sigma, rho, beta)

# Generate 20 random starting points
np.random.seed(42)  # for reproducibility
starting_points = np.random.uniform(-20, 20, (20, 3))

# Colors for each fixed point
colors = ['blue', 'red', 'green']
labels = ['Origin', 'C+', 'C-']

for start_point in starting_points:
    x0, y0, z0 = start_point
    
    # Calculate trajectories for each method
    _, x_euler, y_euler, z_euler = euler_lorenz(x0, y0, z0, t0, tf, dt, sigma, rho, beta)
    _, x_rk, y_rk, z_rk = runge_kutta_lorenz(x0, y0, z0, t0, tf, dt, sigma, rho, beta)
    sol = solve_ivp(lorenz, t_span, [x0, y0, z0], args=(sigma, rho, beta), t_eval=t_eval, method='RK45')
    x_de, y_de, z_de = sol.y

    # Determine which fixed point the trajectory approaches
    end_point = np.array([x_de[-1], y_de[-1], z_de[-1]])
    distances = [np.linalg.norm(end_point - np.array(fp)) for fp in fixed_points]
    closest_fp = np.argmin(distances)
    
    # Plot the trajectory with the color of the closest fixed point
    ax.plot(x_de, y_de, z_de, linewidth=1, color=colors[closest_fp], alpha=0.5)

# Plot fixed points
for i, (x, y, z) in enumerate(fixed_points):
    ax.scatter(x, y, z, color=colors[i], marker='x', s=50, label=f'{labels[i]} ({x:.2f}, {y:.2f}, {z:.2f})')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.figure.set_dpi(300)

ax.legend(loc='upper right')
plt.title(f'Trajectories (ρ = {rho})')
plt.show()