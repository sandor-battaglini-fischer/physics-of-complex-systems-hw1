import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import correlate


sigma = 10
b = 8 / 3
rho = 400

def lorenz_system(state, t):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - b * z
    return [dx_dt, dy_dt, dz_dt]

def integrate_lorenz(initial_state, t):
    return odeint(lorenz_system, initial_state, t)

initial_state = [1.0, 1.0, 1.0]  
t_max = 1000  
dt = 0.01 
t = np.arange(0, t_max, dt)  

trajectory = integrate_lorenz(initial_state, t)

def plot_3d_trajectory(trajectory):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='blue', linewidth=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('Lorenz System Trajectory (Limit Cycle)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], color='blue', linewidth=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

    plt.xlim(-200, 200)
    plt.ylim(-400, 400)

    # plt.axis('equal')

    plt.show()


def plot_time_series(t, trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    axs[0].plot(t, x, color='red')
    axs[0].set_ylabel('X(t)')
    axs[0].grid(True)
    
    axs[1].plot(t, y, color='green')
    axs[1].set_ylabel('Y(t)')
    axs[1].grid(True)
    
    axs[2].plot(t, z, color='blue')
    axs[2].set_xlabel('Time (t)')
    axs[2].set_ylabel('Z(t)')
    axs[2].grid(True)
    
    plt.suptitle('Time Series of Lorenz System Variables')
    plt.show()
    
plot_time_series(t, trajectory)
plot_3d_trajectory(trajectory)



def plot_autocorrelation(t, trajectory, max_lag=1000):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    
    def autocorr(signal, max_lag):
        result = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
        result = result[result.size // 2:]
        return result[:max_lag]
    
    lags = np.arange(max_lag)
    autocorr_x = autocorr(x, max_lag)
    autocorr_y = autocorr(y, max_lag)
    autocorr_z = autocorr(z, max_lag)
    
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    axs[0].plot(lags, autocorr_x, color='red')
    axs[0].set_ylabel('Autocorr X')
    axs[0].grid(True)
    
    axs[1].plot(lags, autocorr_y, color='green')
    axs[1].set_ylabel('Autocorr Y')
    axs[1].grid(True)
    
    axs[2].plot(lags, autocorr_z, color='blue')
    axs[2].set_xlabel('Lag')
    axs[2].set_ylabel('Autocorr Z')
    axs[2].grid(True)
    
    plt.suptitle('Autocorrelation of Lorenz System Variables')
    plt.show()



plot_autocorrelation(t, trajectory)