import pandas as pd
import os
import numpy as np


root_dir = os.path.join(os.path.dirname(__file__), '..')

# Load and set demographic data.
demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')
country_population = (demographics_df['Population'].values[demographics_df.index[demographics_df
                                                                                 ['Country'] == 'Brazil']])

# Load and set time series data.
original_time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')
original_total_cases_time_series = original_time_series_df['Total Cases'].to_numpy()
original_active_cases_time_series = original_time_series_df['Active Cases'].to_numpy()
original_total_deaths_time_series = original_time_series_df['Total Deaths'].to_numpy()
country_population_vector = np.repeat(country_population, len(original_time_series_df))
original_susceptible_population_time_series = country_population_vector - \
                                              original_time_series_df['Total Cases'].to_numpy()
original_time_series_df.insert(len(original_time_series_df.columns), 'Susceptible Population',
                               original_susceptible_population_time_series)
original_daily_new_deaths_time_series = original_time_series_df['Daily New Deaths'].to_numpy()

# Adjust data to reflect real conditions in accordance with recent estimates of under notifications.
adjusted_time_series_df = pd.DataFrame()

adjusted_time_series_df.insert(len(adjusted_time_series_df.columns), 'Total Deaths', original_total_deaths_time_series)

adjusted_time_series_df.insert(len(adjusted_time_series_df.columns), 'Daily New Deaths',
                               original_time_series_df['Daily New Deaths'].to_numpy())

# According to Wu et.al (2020) - Estimating clinical severity of COVID-19 from the transmission dynamics in Wuhan, China
# the real mortality rate is 1.4% and the mean time between disease onset and death is 20 days.
estimated_death_rate = edr = 0.014
mean_time_between_disease_onset_and_death = mtod = 20

adjusted_new_daily_cases_time_series = np.zeros(len(original_time_series_df))
adjusted_total_cases_time_series = np.zeros(len(original_time_series_df))

for i in range(len(original_daily_new_deaths_time_series )):
    if original_daily_new_deaths_time_series[i] == 0:
        pass
    else:
        corresponding_new_cases = original_daily_new_deaths_time_series [i] // edr
        adjusted_new_daily_cases_time_series[i - mtod] = corresponding_new_cases

for i in range(len(adjusted_new_daily_cases_time_series)):
    if np.sum(adjusted_new_daily_cases_time_series[:i]) != 0 and \
            np.sum(adjusted_new_daily_cases_time_series[:i]) == adjusted_total_cases_time_series[i - 1]:
        break
    else:
        adjusted_total_cases_time_series[i] = np.sum(adjusted_new_daily_cases_time_series[:i])


total_cases_growth_rate_array = []

first_non_zero_counted = False

for i in range(len(adjusted_total_cases_time_series)):
    if adjusted_total_cases_time_series[i] == 0:
        pass
    else:
        if first_non_zero_counted == False:
            first_non_zero_counted = True
        else:
            if adjusted_total_cases_time_series[i] != 0:
                total_cases_growth_rate_array.append(
                    adjusted_total_cases_time_series[i]/adjusted_total_cases_time_series[i - 1] - 1)
            else:
                break

total_cases_growth_rate_array = np.array(total_cases_growth_rate_array)

# Remove outlier from first notification.
total_cases_growth_rate_array = total_cases_growth_rate_array[1:]

# Forecast "future" growth rates with exponential fit.
X = np.array(range(len(total_cases_growth_rate_array)) + np.repeat(1, len(total_cases_growth_rate_array)))
Y = total_cases_growth_rate_array

forecast_params = np.polyfit(X, np.log(Y), 1)

forecasted_growth_rates_until_today = np.zeros(mtod-1)

for i in range(mtod - 1):
    forecasted_growth_rates_until_today[i] = np.exp(forecast_params[1]) * np.exp(forecast_params[0] * (i + len(total_cases_growth_rate_array) + 1))

# Forecast "future" total_cases with the growth rates estimates.
adjusted_total_cases_time_series_non_zero_index_list = np.nonzero(adjusted_total_cases_time_series)

last_non_zero_total_case_value = adjusted_total_cases_time_series[adjusted_total_cases_time_series_non_zero_index_list[0][-1]]

forecasted_total_cases_until_today = np.zeros(mtod-1)

forecasted_total_cases_until_today[0] = last_non_zero_total_case_value * (1 + forecasted_growth_rates_until_today[0])

for i in range(mtod - 2):
    forecasted_total_cases_until_today[i + 1] = forecasted_total_cases_until_today[i] * \
                                                (1 + forecasted_growth_rates_until_today[i + 1])

adjusted_total_cases_time_series = np.concatenate((adjusted_total_cases_time_series
                                                   [:len(adjusted_total_cases_time_series) - mtod + 1],
                                                   forecasted_total_cases_until_today))

adjusted_total_cases_time_series = np.ndarray.astype(adjusted_total_cases_time_series, int)

# Insert adjusted total cases on adjusted dataframe.
adjusted_time_series_df.insert(len(adjusted_time_series_df.columns), 'Total Cases', adjusted_total_cases_time_series)

# We are considering that 80% of cases are mild, with recuperation in 2 14 days, while the remaining 18.6% are grave and
# take 21 days to recover.

mild_cases_proportion = 0.8
grave_cases_proportion = 1 - mild_cases_proportion - edr
mild_cases_recovery_time = 14
grave_cases_recovery_time = 21

adjusted_total_recovered_cases_time_series = np.zeros(len(adjusted_time_series_df))

for i in range(21, len(adjusted_time_series_df)):
    adjusted_total_recovered_cases_time_series[i] = \
        adjusted_total_cases_time_series[i - mild_cases_recovery_time] * mild_cases_proportion + \
        adjusted_total_cases_time_series[i - grave_cases_recovery_time] * grave_cases_proportion

adjusted_total_recovered_cases_time_series = np.ndarray.astype(adjusted_total_recovered_cases_time_series, int)

# Insert adjusted total recovery cases on adjusted dataframe.
adjusted_time_series_df.insert(len(adjusted_time_series_df.columns), 'Total Recovery Cases', adjusted_total_recovered_cases_time_series)

adjusted_active_cases_time_series = adjusted_total_cases_time_series - adjusted_total_recovered_cases_time_series - original_total_deaths_time_series

# Insert adjusted total recovery cases on adjusted dataframe.
adjusted_time_series_df.insert(len(adjusted_time_series_df.columns), 'Active Cases', adjusted_active_cases_time_series)

adjusted_time_series_df.to_csv(root_dir + '/data/countries_time_series/brazil_updated.csv', index=False)

