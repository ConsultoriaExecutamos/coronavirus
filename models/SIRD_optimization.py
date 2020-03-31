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
        demographics_df.index[demographics_df['Country'] == 'Brazil']])

    # load and set time series data

    time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')

    total_cases_time_series = time_series_df['Total Cases'].to_numpy()

    active_cases_time_series = time_series_df['Active Cases'].to_numpy()

    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

    total_recover_cases_time_series = pd.DataFrame(
        total_cases_time_series - active_cases_time_series - total_deaths_time_series)

    time_series_df.insert(len(time_series_df.columns), 'Total Recover Cases', total_recover_cases_time_series)

    susceptible_population_time_series = np.repeat(country_population, len(time_series_df))

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

    total_cases = [time_series_df['Total Cases'].values[df_index]]

    SIRD_forecast_time_series = pd.DataFrame({'S': S, 'I': I, 'R': R, 'D': D}, index=[0])

    # model

    forecast_days = periods

    for day in range(1, forecast_days):
        # disease params
        alpha = X[0 + 3*day]
        beta = X[1 + 3*day]
        gamma = X[2 + 3*day]

        last_period_S = SIRD_forecast_time_series['S'].values[len(SIRD_forecast_time_series) - 1]
        last_period_I = SIRD_forecast_time_series['I'].values[len(SIRD_forecast_time_series) - 1]
        last_period_R = SIRD_forecast_time_series['R'].values[len(SIRD_forecast_time_series) - 1]
        last_period_D = SIRD_forecast_time_series['D'].values[len(SIRD_forecast_time_series) - 1]

        susceptible_population_forecast = int(last_period_S - ((alpha / country_population) * last_period_S * last_period_I))
        infected_population_forecast = int(last_period_I + ((alpha / country_population) * last_period_S * last_period_I) - (last_period_I * (beta + gamma)))
        recovered_population_forecast = int(last_period_R + (beta * last_period_I))
        dead_population_forecast = int(last_period_D + (gamma * last_period_I))

        total_cases.append(total_cases[-1] + ((alpha / country_population) * last_period_S * last_period_I))

        SIRD_forecast_time_series = SIRD_forecast_time_series.append(pd.DataFrame({"S": susceptible_population_forecast,
                                                                                   "I": infected_population_forecast,
                                                                                   "R": recovered_population_forecast,
                                                                                   "D": dead_population_forecast},
                                                                                  index=[day]))

    actual_susceptible_population_vector = time_series_df['Active Cases'].values[df_index:]

    forecasted_susceptible_population_vector = SIRD_forecast_time_series['I'].to_numpy()

    actual_infected_vector = time_series_df['Active Cases'].values[df_index:]

    forecasted_infecter_vector = SIRD_forecast_time_series['I'].to_numpy()

    actual_recovered_vector = time_series_df['Total Recover Cases'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_recovered_vector = SIRD_forecast_time_series['R'].to_numpy()

    actual_deaths_vector = time_series_df['Total Deaths'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_desths_vector = SIRD_forecast_time_series['D'].to_numpy()

    plt.plot(actual_susceptible_population_vector, label='Mortes - Atuais')
    plt.plot(forecasted_susceptible_population_vector, label='Mortes - Previsão')
    plt.plot(actual_infected_vector, label='Casos Ativos - Atuais')
    plt.plot(forecasted_infecter_vector, label='Casos Ativos - Previsão')
    plt.plot(actual_recovered_vector, label='Recuperados - Atuais')
    plt.plot(forecasted_recovered_vector, label='Recuperados - Previsão')
    plt.plot(actual_deaths_vector, label='Mortes - Atuais')
    plt.plot(forecasted_desths_vector, label='Mortes - Previsão')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def sird_model_fitting(X):
    root_dir = os.path.join(os.path.dirname(__file__), '..')

    # load and set demographic data
    demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')

    country_population = (demographics_df['Population'].values[
        demographics_df.index[demographics_df['Country'] == 'Brazil']])

    # load and set time series data
    time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')

    total_cases_time_series = time_series_df['Total Cases'].to_numpy()

    active_cases_time_series = time_series_df['Active Cases'].to_numpy()

    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

    total_recover_cases_time_series = pd.DataFrame(
        total_cases_time_series - active_cases_time_series - total_deaths_time_series)

    time_series_df.insert(len(time_series_df.columns), 'Total Recover Cases', total_recover_cases_time_series)

    susceptible_population_time_series = np.repeat(country_population, len(time_series_df))

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

    total_cases = [time_series_df['Total Cases'].values[df_index]]

    SIRD_forecast_time_series = pd.DataFrame({'S': S, 'I': I, 'R': R, 'D': D}, index=[0])

    # model

    forecast_days = len(time_series_df) - df_index - 1

    for day in range(1, forecast_days):
        # disease params
        alpha = X[0 + 3 * day]
        beta = X[1 + 3 * day]
        gamma = X[2 + 3 * day]

        last_period_S = SIRD_forecast_time_series['S'].values[len(SIRD_forecast_time_series) - 1]
        last_period_I = SIRD_forecast_time_series['I'].values[len(SIRD_forecast_time_series) - 1]
        last_period_R = SIRD_forecast_time_series['R'].values[len(SIRD_forecast_time_series) - 1]
        last_period_D = SIRD_forecast_time_series['D'].values[len(SIRD_forecast_time_series) - 1]

        susceptible_population_forecast = int(last_period_S - ((alpha / country_population) * last_period_S * last_period_I))
        infected_population_forecast = int(last_period_I + ((alpha / country_population) * last_period_S * last_period_I) - (last_period_I * (beta + gamma)))
        recovered_population_forecast = int(last_period_R + (beta * last_period_I))
        dead_population_forecast = int(last_period_D + (gamma * last_period_I))

        total_cases.append(total_cases[-1] + ((alpha / country_population) * last_period_S * last_period_I))



        SIRD_forecast_time_series = SIRD_forecast_time_series.append(pd.DataFrame({"S": susceptible_population_forecast,
                                                                                   "I": infected_population_forecast,
                                                                                   "R": recovered_population_forecast,
                                                                                   "D": dead_population_forecast},
                                                                                  index=[day]))

    actual_susceptible_population_vector = time_series_df['Total Deaths'].values[df_index: df_index + len(SIRD_forecast_time_series)]

    forecasted_susceptible_population_vector = SIRD_forecast_time_series['D'].to_numpy()

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

    root_dir = os.path.join(os.path.dirname(__file__), '..')

    # load and set demographic data
    demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')

    country_population = (demographics_df['Population'].values[
        demographics_df.index[demographics_df['Country'] == 'Brazil']])

    # load and set time series data
    time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')

    total_cases_time_series = time_series_df['Total Cases'].to_numpy()

    active_cases_time_series = time_series_df['Active Cases'].to_numpy()

    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

    total_recover_cases_time_series = pd.DataFrame(
            total_cases_time_series - active_cases_time_series - total_deaths_time_series)

    time_series_df.insert(len(time_series_df.columns), 'Total Recover Cases', total_recover_cases_time_series)

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

    forecast_days = len(time_series_df) - df_index - 1

    bounds = []

    alpha_range = (0, 1)
    beta_range = (0, 1)
    gama_range = (0, 0.1)

    for i in range(forecast_days):
        bounds.append(alpha_range)
        bounds.append(beta_range)
        bounds.append(gama_range)

    optimized_SIRD_model_result = differential_evolution(sird_model_fitting, bounds=bounds, workers=16)

    parameters_list = optimized_SIRD_model_result.x

    R_naught_list = []

    for day in range(forecast_days):

        alpha = parameters_list[0 + 3 * day]
        beta = parameters_list[1 + 3 * day]
        gamma = parameters_list[2 + 3 * day]

        R_naught = alpha / (beta + gamma)

        R_naught_list.append(R_naught)

    print(parameters_list)

    print(len(parameters_list))

    print(R_naught_list)

    periods_for_forecasting = 20

    last_alpha = parameters_list[-3]
    last_beta = parameters_list[-2]
    last_gamma = parameters_list[-1]

    print(forecast_days)

    print(periods_for_forecasting - forecast_days)

    for i in range(periods_for_forecasting - forecast_days):
        parameters_list = np.append(parameters_list, parameters_list[-3:])

    print(parameters_list)

    print(len(parameters_list))

    sird_model(parameters_list, periods=periods_for_forecasting)



