import numpy as np
import pandas as pd
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm, t, gompertz
from scipy.optimize import differential_evolution


# Define funcion with the coefficients to estimate
def func_logistic(t, a, b, c):
    return c / (1 + a * np.exp(-b*t))


def func_sin(t, a, b, c, d):
    return a*np.sin(b*(t - c)) + d


def cdf(t, a, b):
    return norm.cdf(t, a, b)


def cdf_student(t, a, b, c):
    return t.cdf(t, a, b, c)


def gompertz_model(X, x):
    #a = upper asymptote
    #b = negative = x axis displacement
    #c = negative = growth rate
    return X[0]*(np.exp(X[1]*(np.exp(X[2]*x))))


def cost_analysis(X, *args):
    x = args[0]
    y = args[1]
    forecast_vector = np.zeros(len(x))

    sum_of_square_errors = 0

    for i in range(len(x)):
        forecast_vector[i] = gompertz_model(X, x[i])

    squared_errors = np.square(forecast_vector - y)

    sum_of_square_errors += np.sum(squared_errors)

    return sum_of_square_errors

########################################################################################################################
########################################################################################################################


root_dir = os.path.join(os.path.dirname(__file__), '..')

# Load and set time series data.
time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')
total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

dead_var = 0
df_index = 0
while dead_var == 0:
    dead_var = time_series_df['Total Deaths'].values[df_index]
    df_index += 1

df_index -= 1

total_deaths_time_series = total_deaths_time_series[df_index:]

time_steps = range(len(total_deaths_time_series))

time_steps = np.array(time_steps)

# Convert pd.Series to np.Array and use Scipy's curve fit to find   # the best Nonlinear Least Squares coefficients
x = time_steps
y = total_deaths_time_series

optimal_params = differential_evolution(cost_analysis, args=(x, y,), bounds=[(-10, 60000),
                                                                             (-20, 100),
                                                                             (-10, 100)],
                                        disp=True, polish=True, tol=0.01, maxiter=1000, seed=0)

print(optimal_params.x)

x = range(150)

x = np.array(x)

y_hat = np.zeros(len(x))

for i in range(len(x)):
    y_hat[i] = gompertz_model(optimal_params.x, i)

plt.plot(y_hat, 'r-', y, 'o')
plt.show()

