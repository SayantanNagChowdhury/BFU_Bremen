# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:15:27 2024

@author: snagchowdh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
alpha = 60.0
beta = 0.01
K = 0.02
gamma = 0.01

# Define the system of differential equations
def dZdt(t, Z):
    y1, y2 = Z
    dy1dt = alpha / (1 + y2 / K) * (beta + y1**2) / (1 + y1**2) - y1
    dy2dt = gamma * (y1 - y2)
    return [dy1dt, dy2dt]

# Initial conditions
y1_0 = np.random.rand()  # Random initial condition for z_r
y2_0 = np.random.rand()  # Random initial condition for z_i
Z0 = [y1_0, y2_0]

# Time parameters
t_span = (0, 2000)  # Time span
t_eval = np.arange(1500, 2000, 0.01)  # Time points where the solution is desired (last 500 units)

# Solve the initial value problem
sol = solve_ivp(dZdt, t_span, Z0, method='RK45', t_eval=t_eval)

# Plotting the results
plt.plot(sol.t, sol.y[0], label='z_r')
plt.plot(sol.t, sol.y[1], label='z_i')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Solution of the Differential Equations with Random Initial Conditions (Last 500 Units)')
plt.legend()
plt.grid(True)
plt.show()


# # Calculate the amplitude of the oscillator
# amplitude = np.sqrt(sol.y[0]**2 + sol.y[1]**2)

# # Plotting the amplitude of the oscillator
# plt.plot(sol.t, amplitude, label='Amplitude')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Amplitude of the Oscillator with Random Initial Conditions (Last 500 Units)')
# plt.legend()
# plt.grid(True)
# plt.show()