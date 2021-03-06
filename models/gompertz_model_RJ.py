import numpy as np
import pandas as pd
import os
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm, t, gompertz
from scipy.optimize import differential_evolution


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
time_series_df = pd.read_csv(root_dir + '/data/brazillian_states_time_series/RJ.csv')
total_deaths_time_series = time_series_df['dead'].to_numpy()

total_deaths_time_series = np.flip(total_deaths_time_series)

dead_var = 0
df_index = 0
while dead_var == 0:
    dead_var = total_deaths_time_series[df_index]
    df_index += 1

df_index -= 1

total_deaths_time_series = total_deaths_time_series[df_index:]

time_steps = range(len(total_deaths_time_series))

time_steps = np.array(time_steps)

x = time_steps
y = total_deaths_time_series

optimal_params = differential_evolution(cost_analysis, args=(x, y,), bounds=[(-10, 2200),
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

