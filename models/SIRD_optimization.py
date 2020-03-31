import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.optimize import differential_evolution


def sird_model(X, periods=15):
    root_dir = os.path.join(os.path.dirname(__file__), '..')

    # load and set demographic data
    demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')

    country_population = (demographics_df['Population'].values[
        demographics_df.index[demographics_df['Country'] == 'Us']])

    # load and set time series data

    time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/us.csv')

    total_cases_time_series = time_series_df['Total Cases'].to_numpy()

    active_cases_time_series = time_series_df['Active Cases'].to_numpy()

    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

    total_recover_cases_time_series = pd.DataFrame(
        total_cases_time_series - active_cases_time_series - total_deaths_time_series)

    time_series_df.insert(len(time_series_df.columns), 'Total Recover Cases', total_recover_cases_time_series)

    country_population_vector = np.repeat(country_population, len(time_series_df))

    susceptible_population_time_series = country_population_vector - time_series_df['Total Cases'].to_numpy()

    time_series_df.insert(len(time_series_df.columns), 'Susceptible Population', susceptible_population_time_series)

    # data retrieval from dataframes

    infected_var = 0
    recovered_var = 0
    dead_var = 0

    df_index = 0
    while infected_var == 0 or recovered_var == 0 or dead_var == 0:
        infected_var = time_series_df['Active Cases'].values[df_index]
        recovered_var = time_series_df['Total Recover Cases'].values[df_index]
        dead_var = time_series_df['Total Deaths'].values[df_index]
        df_index += 1

    susceptible_population = time_series_df['Susceptible Population'].values[df_index]

    current_infected_population = time_series_df['Active Cases'].values[df_index]

    recovered_population = time_series_df['Total Recover Cases'].values[df_index]

    dead_population = time_series_df['Total Deaths'].values[df_index]

    # initial model values
    S = susceptible_population  # susceptible population
    I = current_infected_population  # infected population
    R = recovered_population  # recovered population
    D = dead_population  # dead population

    total_cases_forecast = [time_series_df['Total Cases'].values[df_index]]

    SIRD_forecast_time_series = pd.DataFrame({'S': S, 'I': I, 'R': R, 'D': D}, index=[0])

    # model

    forecast_days = periods

    for day in range(1, forecast_days):
        # disease params
        alpha = X[0]
        beta = X[1]
        gamma = X[2]

        last_period_S = SIRD_forecast_time_series['S'].values[len(SIRD_forecast_time_series) - 1]
        last_period_I = SIRD_forecast_time_series['I'].values[len(SIRD_forecast_time_series) - 1]
        last_period_R = SIRD_forecast_time_series['R'].values[len(SIRD_forecast_time_series) - 1]
        last_period_D = SIRD_forecast_time_series['D'].values[len(SIRD_forecast_time_series) - 1]

        susceptible_population_forecast = int(last_period_S - ((alpha / country_population) * last_period_S * last_period_I))
        infected_population_forecast = int(last_period_I + ((alpha / country_population) * last_period_S * last_period_I) - (last_period_I * (beta + gamma)))
        recovered_population_forecast = int(last_period_R + (beta * last_period_I))
        dead_population_forecast = int(last_period_D + (gamma * last_period_I))

        total_cases_forecast.append(total_cases_forecast[-1] + ((alpha / country_population) * last_period_S * last_period_I))

        SIRD_forecast_time_series = SIRD_forecast_time_series.append(pd.DataFrame({"S": susceptible_population_forecast,
                                                                                   "I": infected_population_forecast,
                                                                                   "R": recovered_population_forecast,
                                                                                   "D": dead_population_forecast},
                                                                                  index=[day]))

    actual_susceptible_population_vector = time_series_df['Susceptible Population'].values[df_index:]

    forecasted_susceptible_population_vector = SIRD_forecast_time_series['S'].to_numpy()

    actual_infected_vector = time_series_df['Active Cases'].values[df_index:]

    forecasted_infecter_vector = SIRD_forecast_time_series['I'].to_numpy()

    actual_recovered_vector = time_series_df['Total Recover Cases'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_recovered_vector = SIRD_forecast_time_series['R'].to_numpy()

    actual_deaths_vector = time_series_df['Total Deaths'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_deaths_vector = SIRD_forecast_time_series['D'].to_numpy()

    actual_total_cases_vector = time_series_df['Total Cases'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    print(actual_total_cases_vector)

    forecasted_total_cases_vector = total_cases_forecast

    # plt.plot(actual_susceptible_population_vector, label='Susceptible Population - Actual')
    # plt.plot(forecasted_susceptible_population_vector, label='Susceptible Population - Forecast')
    plt.plot(actual_infected_vector, label='Active Cases - Actual')
    plt.plot(forecasted_infecter_vector, label='Active Cases - Forecast')
    plt.plot(actual_recovered_vector, label='Recovered - Actual')
    plt.plot(forecasted_recovered_vector, label='Recovered - Forecast')
    plt.plot(actual_deaths_vector, label='Deaths - Actual')
    plt.plot(forecasted_deaths_vector, label='Deaths - Forecast')
    # plt.plot(actual_total_cases_vector, label='Total Cases - Actual')
    # plt.plot(forecasted_total_cases_vector, label='Total Cases - Forecast')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def sird_model_fitting(X):
    root_dir = os.path.join(os.path.dirname(__file__), '..')

    # load and set demographic data
    demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')

    country_population = (demographics_df['Population'].values[
        demographics_df.index[demographics_df['Country'] == 'Us']])

    # load and set time series data
    time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/us.csv')

    total_cases_time_series = time_series_df['Total Cases'].to_numpy()

    active_cases_time_series = time_series_df['Active Cases'].to_numpy()

    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

    total_recover_cases_time_series = pd.DataFrame(
        total_cases_time_series - active_cases_time_series - total_deaths_time_series)

    time_series_df.insert(len(time_series_df.columns), 'Total Recover Cases', total_recover_cases_time_series)

    country_population_vector = np.repeat(country_population, len(time_series_df))

    susceptible_population_time_series = country_population_vector - time_series_df['Total Cases'].to_numpy()

    time_series_df.insert(len(time_series_df.columns), 'Susceptible Population', susceptible_population_time_series)

    # data retrieval from dataframes

    infected_var = 0
    recovered_var = 0
    dead_var = 0

    df_index = 0
    while infected_var == 0 or recovered_var == 0 or dead_var == 0:
        infected_var = time_series_df['Active Cases'].values[df_index]
        recovered_var = time_series_df['Total Recover Cases'].values[df_index]
        dead_var = time_series_df['Total Deaths'].values[df_index]
        df_index += 1

    susceptible_population = time_series_df['Susceptible Population'].values[df_index]

    current_infected_population = time_series_df['Active Cases'].values[df_index] * 20

    recovered_population = time_series_df['Total Recover Cases'].values[df_index] * 40

    dead_population = time_series_df['Total Deaths'].values[df_index]

    # initial model values
    S = susceptible_population  # susceptible population
    I = current_infected_population  # infected population
    R = recovered_population  # recovered population
    D = dead_population  # dead population

    total_cases_forecast = [time_series_df['Total Cases'].values[df_index]]

    SIRD_forecast_time_series = pd.DataFrame({'S': S, 'I': I, 'R': R, 'D': D}, index=[0])

    # model

    forecast_days = len(time_series_df) - df_index - 1

    for day in range(1, forecast_days):
        # disease params
        alpha = X[0]
        beta = X[1]
        gamma = X[2]

        last_period_S = SIRD_forecast_time_series['S'].values[len(SIRD_forecast_time_series) - 1]
        last_period_I = SIRD_forecast_time_series['I'].values[len(SIRD_forecast_time_series) - 1]
        last_period_R = SIRD_forecast_time_series['R'].values[len(SIRD_forecast_time_series) - 1]
        last_period_D = SIRD_forecast_time_series['D'].values[len(SIRD_forecast_time_series) - 1]

        susceptible_population_forecast = int(last_period_S - ((alpha / country_population) * last_period_S * last_period_I))
        infected_population_forecast = int(last_period_I + ((alpha / country_population) * last_period_S * last_period_I) - (last_period_I * (beta + gamma)))
        recovered_population_forecast = int(last_period_R + (beta * last_period_I))
        dead_population_forecast = int(last_period_D + (gamma * last_period_I))

        total_cases_forecast.append(country_population - susceptible_population)


        SIRD_forecast_time_series = SIRD_forecast_time_series.append(pd.DataFrame({"S": susceptible_population_forecast,
                                                                                   "I": infected_population_forecast,
                                                                                   "R": recovered_population_forecast,
                                                                                   "D": dead_population_forecast},
                                                                                  index=[day]))


    actual_susceptible_population_vector = time_series_df['Susceptible Population'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_susceptible_population_vector = SIRD_forecast_time_series['S'].to_numpy()

    squared_susceptible_population_errors = np.square(forecasted_susceptible_population_vector - actual_susceptible_population_vector)

    sum_of_susceptible_population_squared_errors = np.sum(squared_susceptible_population_errors)


    actual_infected_vector = time_series_df['Active Cases'].values[df_index : df_index + len(SIRD_forecast_time_series)]

    forecasted_infecter_vector = SIRD_forecast_time_series['I'].to_numpy()

    infected_squared_errors = np.square(forecasted_infecter_vector - actual_infected_vector)

    sum_of_infected_squared_errors = np.sum(infected_squared_errors)


    actual_recovered_vector = time_series_df['Total Recover Cases'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_recovered_vector = SIRD_forecast_time_series['R'].to_numpy()

    recovered_squared_errors = np.square(forecasted_recovered_vector - actual_recovered_vector)

    sum_of_recovered_squared_errors = np.sum(recovered_squared_errors)


    actual_deaths_vector = time_series_df['Total Deaths'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_deaths_vector = SIRD_forecast_time_series['D'].to_numpy()

    squared_deaths_errors = np.square(forecasted_deaths_vector - actual_deaths_vector)

    sum_of_deaths_squared_errors = np.sum(squared_deaths_errors)


    return sum_of_susceptible_population_squared_errors + sum_of_infected_squared_errors + sum_of_recovered_squared_errors + sum_of_deaths_squared_errors


########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':

    alpha_range = (0, 1)
    beta_range = (0, 1)
    gama_range = (0, 0.1)

    bounds = [alpha_range, beta_range, gama_range]

    optimized_SIRD_model_result = differential_evolution(sird_model_fitting, bounds=bounds, workers=16)

    parameters_list = optimized_SIRD_model_result.x

    R_naught = parameters_list[0] / (parameters_list[1] + parameters_list[2])

    print(R_naught)

    print(parameters_list)

    # parameters_list =

    periods_for_forecasting = 40

    sird_model(parameters_list, periods=periods_for_forecasting)

