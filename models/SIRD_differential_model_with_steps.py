import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import differential_evolution


def SIRD_model(y, t, N, X):
    # SIRD model differential equations.
    S, I, R, D = y
    dSdt = -X[0] * S * I / N
    dIdt = X[0] * S * I / N - (0.1 + 0.00147) * I
    dDdt = 0.1 * I
    dRdt = 0.00147 * I
    return dSdt, dIdt, dRdt, dDdt


def SIRD_model_sim(y0, t, N, X):
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(SIRD_model, y0, t, args=(N, X))
    S, I, R, D = ret.T
    return S, I, R, D


def SIRD_model_fitting(X, *args):
    y0 = args[0]
    N = args[1]
    t = args[2]
    initial_index = args[3]
    final_index = args[4]

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(SIRD_model, y0, t, args=(N, X))

    S, I, R, D = ret.T

    time_series_columns_list = ['Susceptible Population', 'Active Cases', 'Total Recover Cases', 'Total Deaths']
    forecasts_list = [S, I, R, D]

    sum_of_squared_errors = 0

    for i in range(4):
        # Crop time series to retrieve data with at least one infected.
        actual_vector = time_series_df[time_series_columns_list[i]].values[initial_index:final_index]
        forecasted_vector = forecasts_list[i]
        squared_errors = np.square(forecasted_vector - actual_vector)
        sum_of_squared_errors += np.sum(squared_errors)

    return sum_of_squared_errors


def obtain_best_fit_estimators(y0, N, t, bounds, initial_index, final_index):
    optimized_SIRD_model_result = differential_evolution(SIRD_model_fitting, bounds=bounds, args=(y0, N, t,
                                                                                                  initial_index,
                                                                                                  final_index))
    return optimized_SIRD_model_result.x


def plot_SIRD_model(S, I, R, D):
    actual_S_vector = time_series_df['Susceptible Population'].values[df_index: df_index + len(S)]
    actual_I_vector = time_series_df['Active Cases'].values[df_index: df_index + len(I)]
    actual_R_vector = time_series_df['Total Recover Cases'].values[df_index: df_index + len(R)]
    actual_D_vector = time_series_df['Total Deaths'].values[df_index: df_index + len(D)]

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    # ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected - Forecast')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity - Forecast')
    ax.plot(t, D, 'y', alpha=0.5, lw=2, label='Dead - Forecast')
    ax.plot(t, actual_I_vector, 'k', alpha=0.5, lw=2, label='Infected - Actual')
    ax.plot(t, actual_R_vector, 'c', alpha=0.5, lw=2, label='Recovered with immunity - Actual')
    ax.plot(t, actual_D_vector, 'm', alpha=0.5, lw=2, label='Dead - Actual')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    # ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()


def fit_best_parameters_per_step(y0, N, t, bounds, step_size):
    number_of_steps = t_length // step_size
    to_skip = t_length % step_size
    alpha_list = []
    beta_list = []
    gamma_list = []
    for step in range(number_of_steps):
        initial_index = df_index + to_skip + step_size*step
        final_index = initial_index + step_size
        best_fit_estimators = obtain_best_fit_estimators(y0, N, t, bounds, initial_index, final_index)
        alpha_list.append(best_fit_estimators[0])
        # beta_list.append(best_fit_estimators[1])
        # gamma_list.append(best_fit_estimators[2])
    return alpha_list, beta_list, gamma_list

########################################################################################################################
########################################################################################################################


if __name__ == '__main__':

    root_dir = os.path.join(os.path.dirname(__file__), '..')

    # Load and set demographic data.
    demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')
    country_population = (demographics_df['Population'].values[
        demographics_df.index[demographics_df['Country'] == 'Brazil']])

    # Load and set time series data.
    time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')
    total_cases_time_series = time_series_df['Total Cases'].to_numpy()
    active_cases_time_series = time_series_df['Active Cases'].to_numpy()
    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()
    dates_time_series = time_series_df['Date'].to_numpy()
    total_recover_cases_time_series = pd.DataFrame(
        total_cases_time_series - active_cases_time_series - total_deaths_time_series)
    time_series_df.insert(len(time_series_df.columns), 'Total Recover Cases', total_recover_cases_time_series)
    country_population_vector = np.repeat(country_population, len(time_series_df))
    susceptible_population_time_series = country_population_vector - time_series_df['Total Cases'].to_numpy()
    time_series_df.insert(len(time_series_df.columns), 'Susceptible Population', susceptible_population_time_series)

    # Data retrieval from dataframes.
    infected_var = 0
    recovered_var = 0
    dead_var = 0

    df_index = 0
    while infected_var == 0 and recovered_var == 0 and dead_var == 0:
        infected_var = time_series_df['Active Cases'].values[df_index]
        recovered_var = time_series_df['Total Recover Cases'].values[df_index]
        dead_var = time_series_df['Total Deaths'].values[df_index]
        df_index += 1

    susceptible_population = time_series_df['Susceptible Population'].values[df_index]
    current_infected_population = time_series_df['Active Cases'].values[df_index]
    recovered_population = time_series_df['Total Recover Cases'].values[df_index]
    dead_population = time_series_df['Total Deaths'].values[df_index]

    # Initial model values.
    S = susceptible_population  # susceptible population
    I = current_infected_population  # infected population
    R = recovered_population  # recovered population
    D = dead_population  # dead population

    y0 = S, I, R, D

    N = country_population.item(0)

    alpha_bounds = (0, 1)
    beta_bounds = (0, 1)
    gamma_bounds = (0, 0.1)

    # bounds = [alpha_bounds, beta_bounds, gamma_bounds]
    bounds = [alpha_bounds]

    step_size = 3

    # Determine time grid for fitting
    t_length = len(time_series_df) - df_index - 1

    t = t = np.linspace(0, step_size, step_size)

    best_fit_estimators_list = fit_best_parameters_per_step(y0, N, t, bounds, step_size)

    print(best_fit_estimators_list)

    # t = np.linspace(0, len(best_fit_estimators_list[0]), len(best_fit_estimators_list[0]))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(111, axisbelow=True)
    # ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(t, np.array(best_fit_estimators_list[0]), 'r', alpha=0.5, lw=2, label='Alpha')
    # ax.plot(t, np.array(best_fit_estimators_list[1]), 'g', alpha=0.5, lw=2, label='Beta')
    # ax.plot(t, np.array(best_fit_estimators_list[2]), 'y', alpha=0.5, lw=2, label='Gamma')

    # ax.plot(dates_time_series, adjusted_total_cases_time_series, 'g', alpha=0.5, lw=2, label='Adjusted New Infetions')

    # ax.set_xlabel('Time /days')
    # ax.set_ylabel('Number (1000s)')
    # ax.set_ylim(0,1.2)
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # for spine in ('top', 'right', 'bottom', 'left'):
    #     ax.spines[spine].set_visible(False)
    # plt.show()

    # S, I, R, D = SIRD_model_sim(y0, t, N, best_fit_estimators)
    #
    # plot_SIRD_model(S, I, R, D)

