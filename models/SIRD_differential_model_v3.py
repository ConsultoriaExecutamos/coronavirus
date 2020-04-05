import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import differential_evolution

def SIRD_model(y, t, N, X):
    # SIRD model differential equations.
    S, IN, IA, IS, R, D, SUM = y
    dSdt = (-X[0]*S*(IA/(S+IA+IS+IN+R)))
    dINdt = (X[0]*S*(IA/N)) - (X[3] * IN)
    dIAdt = X[3] * IN - (X[4] * IA)
    dISdt = X[4] * IA - (X[1] + X[2]) * IS
    dRdt = X[1] * IS
    dDdt = X[2] * IS
    dSUMdt = (X[0]*S*(IA/N))
    return dSdt, dINdt, dIAdt, dISdt, dRdt, dDdt, dSUMdt


def SIRD_model_sim(y0, t, N, X):
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(SIRD_model, y0, t, args=(N, X))
    S, IN, IA, IS, R, D, SUM = ret.T
    return S, IN, IA, IS, R, D, SUM


def SIRD_model_fitting(X, *args):
    y0 = args[0]
    N = args[1]
    t = args[2]

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(SIRD_model, y0, t, args=(N, X))

    S, IN, IA, IS, R, D, SUM = ret.T

    # time_series_columns_list = ['Active Cases', 'Total Recovered Cases', 'Total Deaths', 'Total Cases']
    # forecasts_list = [[IN + IA + IS], R, D, SUM]

    time_series_columns_list = ['Active Cases','Total Recovery Cases', 'Total Deaths']
    forecasts_list = [[IN + IA + IS],R, D]

    sum_of_squared_errors = 0

    for i in range(len(forecasts_list)):
        # Crop time series to retrieve data with at least one infected.
        actual_vector = time_series_df[time_series_columns_list[i]].values[11:]
        forecasted_vector = forecasts_list[i]
        actual_vector+=1
        squared_errors = np.abs((forecasted_vector/actual_vector)-1)
        sum_of_squared_errors += np.sum(squared_errors)

    return sum_of_squared_errors


def obtain_best_fit_estimators(y0, N, t, bounds):
    optimized_SIRD_model_result = differential_evolution(SIRD_model_fitting, bounds=bounds, args=(y0, N, t,),
                                                         maxiter=500, disp=True, polish=True, mutation=(0,1.9),workers=1)
    return optimized_SIRD_model_result.x


def plot_SIRD_model(S, IN, IA, IS, R, D, SUM,t):
    #actual_I_vector = time_series_df['Active Cases'].values[8:]
    #actual_SUM_vector = time_series_df['Total Cases'].values[8:]
    #actual_RECOVER_vector = time_series_df['Total Recovered Cases'].values[8:]
    #actual_DEATHS_vector = time_series_df['Total Deaths'].values[8:]

    #actual_R_vector = time_series_df['Total Recovered Cases'].values[df_index: df_index + len(R)]
    #actual_D_vector = time_series_df['Total Deaths'].values[df_index: df_index + len(D)]

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    # ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, IN, 'k', alpha=0.5, lw=2, label='Infected Incubating - Forecast')
    ax.plot(t, IA, 'c', alpha=0.5, lw=2, label='Infected Active - Forecast')
    ax.plot(t, IS, 'm', alpha=0.5, lw=2, label='Infected Isolated - Forecast')
    ax.plot(t, IN + IA + IS, 'r', alpha=0.5, lw=2, label='Infected Total - Forecast')
    ax.plot(t, SUM, 'b', alpha=0.5, lw=2, label='Total Cases - Forecast')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity - Forecast')
    ax.plot(t, D, 'y', alpha=0.5, lw=2, label='Dead - Forecast')

    #ax.plot(t, actual_I_vector, 'o', alpha=0.5, lw=2, label='Actual Total Infected')
    #ax.plot(t, actual_SUM_vector, 'o', alpha=0.5, lw=2, label='Actual Total Cases')
    #ax.plot(t, actual_RECOVER_vector, 'o', alpha=0.5, lw=2, label='Actual Recovered')
    #ax.plot(t, actual_DEATHS_vector, 'o', alpha=0.5, lw=2, label='Actual Total Deaths')


    # ax.plot(t, actual_I_vector, 'k', alpha=0.5, lw=2, label='Infected - Actual')
    # ax.plot(t, actual_R_vector, 'c', alpha=0.5, lw=2, label='Recovered with immunity - Actual')
    # ax.plot(t, actual_D_vector, 'm', alpha=0.5, lw=2, label='Dead - Actual')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    # ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()


########################################################################################################################
########################################################################################################################


if __name__ == '__main__':

    root_dir = os.path.join(os.path.dirname(__file__), '..')

    # Load and set demographic data.
    demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')
    country_population = (demographics_df['Population'].values[
        demographics_df.index[demographics_df['Country'] == 'Brazil']])

    # Load and set time series data.
    time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil_updated.csv')
    total_cases_time_series = time_series_df['Total Cases'].to_numpy()
    active_cases_time_series = time_series_df['Active Cases'].to_numpy()
    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()
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
        recovered_var = time_series_df['Total Recovery Cases'].values[df_index]
        dead_var = time_series_df['Total Deaths'].values[df_index]
        df_index += 1

    susceptible_population = time_series_df['Susceptible Population'].values[df_index]
    current_infected_population = time_series_df['Active Cases'].values[df_index]
    recovered_population = time_series_df['Total Recovery Cases'].values[df_index]
    dead_population = time_series_df['Total Deaths'].values[df_index]

    # Initial model values.
    # S = susceptible_population  # susceptible population
    # I = current_infected_population  # infected population
    # R = recovered_population  # recovered population
    # D = dead_population  # dead population
    S0= country_population*0.64*0.5*0.4
    y0 = S0, 80, 40, 0, 0, 0, 0

    N = country_population.item(0)

    # Determine time grid for fitting
    t_length = len(time_series_df) - df_index - 1
    t = t = np.linspace(0, 39, 39)

    alpha_bounds = (1,10)
    beta_bounds = (0.071, 0.14)
    gamma_bounds = (0.00214, 0.015)
    tetta_bound = (0.14, 0.3)
    delta_bound = (0.14, 0.2)

    bounds = [alpha_bounds, beta_bounds, gamma_bounds, tetta_bound, delta_bound]

    best_fit_estimators = obtain_best_fit_estimators(y0, N, t, bounds)

#    params = [4.34320881e+00 1.40000000e-01 2.14000000e-03 3.00000000e-01
    #  2.00000000e-01]
    t = t = np.linspace(0, 120, 120) # projections
    print(best_fit_estimators)
    S, IN, IA, IS, R, D, SUM = SIRD_model_sim(y0, t, N, best_fit_estimators)
    plot_SIRD_model(S, IN, IA, IS, R, D, SUM,t)

    