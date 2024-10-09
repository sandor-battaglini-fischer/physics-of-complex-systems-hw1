# Lorenz System Analysis - Homework 1

This repository contains a collection of Python scripts for analyzing the Lorenz system, a set of differential equations originally developed to model atmospheric convection. The Lorenz system is known for its chaotic solutions and is a classic example of a dynamical system exhibiting chaos.

## Contents

- **calculate_char_poly.py**:
  - Computes the characteristic polynomial of a given matrix.
  - Includes functionality to calculate and display the roots of the polynomial.

- **1-3.py**:
  - Implements Euler and Runge-Kutta methods for solving the Lorenz system.
  - Compares numerical solutions with reference data and calculates errors.
  - Visualizes trajectories and fixed points in 3D space.

- **1-4.py**:
  - Uses the Runge-Kutta method to integrate the Lorenz system.
  - Estimates the maximum Lyapunov exponent to measure the system's sensitivity to initial conditions.

- **1-5.py**:
  - Integrates the Lorenz system and visualizes the trajectory in 3D.
  - Plots time series and autocorrelation of the system's variables.

- **1-7-0-1d.py**:
  - Generates a 1D interval and calculates the correlation dimension using a range of epsilon values.
  - Visualizes the correlation sum and fits a line to estimate the correlation dimension.

- **1_7_1_lorenz.py**:
  - Simulates the Lorenz attractor using the Lorenz system equations.
  - Estimates the correlation dimension using both manual and automatic scaling region fits.
  - Visualizes the correlation integral and local slopes.

## Getting Started

To run the scripts, ensure you have Python installed along with the required libraries. You can install the dependencies using:

```bash
pip install numpy matplotlib scipy sympy scikit-learn
``
