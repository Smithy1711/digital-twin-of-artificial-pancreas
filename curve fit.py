import numpy as np
from scipy.optimize import curve_fit

# Define the function to fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate some sample data
xdata = np.linspace(0, 4, 50)
known_param = 1.3
ydata = func(xdata, 2.5, known_param, 0.5) + 0.2 * np.random.normal(size=len(xdata))

# Define the function to fit with the known parameter fixed
def func_fixed(x, a, c):
    return a * np.exp(-known_param * x) + c

# Fit the data with the fixed parameter
popt, pcov = curve_fit(func_fixed, xdata, ydata, p0=[2, 0.3])

# Print the optimized parameters
print(popt)