import matplotlib.pyplot as plt
import pandas as pd
import os


########################################################################################################################
########################################################################################################################
########################################################################################################################
# load and set data
root_dir = os.path.join(os.path.dirname( __file__ ), '..' )

time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')

total_cases_time_series = time_series_df['Total Cases'].to_numpy()

active_cases_time_series = time_series_df['Active Cases'].to_numpy()

total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

recover_cases_time_series = pd.DataFrame(total_cases_time_series - active_cases_time_series - total_deaths_time_series)

time_series_df.insert(len(time_series_df.columns), 'Recover Cases', recover_cases_time_series)

demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')


# retrieve country data
country_population = demographics_df['Population'].values[demographics_df.index[demographics_df['Country'] == 'Brazil'].tolist()[0]]

current_infected_population = time_series_df['Active Cases'].values[len(time_series_df) - 1]

recovered_population = time_series_df['Recover Cases'].values[len(time_series_df) - 1]

dead_population = time_series_df['Total Deaths'].values[len(time_series_df) - 1]

susceptible_population = country_population - recovered_population - dead_population - current_infected_population


# params estimation




# initial model values
S = susceptible_population          # susceptible population
I = current_infected_population     # infected population
R = recovered_population            # recovered population
D = dead_population                 # dead population

forecast_days = 30

# disease params
alpha = 0.14       # infection rate    - source:
beta = 0.05        # recovery rate     - source:
gamma = 0.035         # fatality rate     - source:


SIRD_forecast_time_series = pd.DataFrame({'S': S, 'I': I, 'R': R, 'D': D}, index=[0])

# model
for day in range(1, forecast_days):
    last_period_S = SIRD_forecast_time_series['S'].values[len(SIRD_forecast_time_series) - 1]
    last_period_I = SIRD_forecast_time_series['I'].values[len(SIRD_forecast_time_series) - 1]
    last_period_R = SIRD_forecast_time_series['R'].values[len(SIRD_forecast_time_series) - 1]
    last_period_D = SIRD_forecast_time_series['D'].values[len(SIRD_forecast_time_series) - 1]

    susceptible_population_forecast = int(last_period_S - ((alpha/country_population)*last_period_S*last_period_I))
    infected_population_forecast = int(last_period_I + ((alpha/country_population)*last_period_S*last_period_I) - (last_period_I*(beta + gamma)))
    recovered_population_forecast = int(last_period_R + (beta*last_period_I))
    dead_population_forecast = int(last_period_D + (gamma*last_period_I))

    SIRD_forecast_time_series = SIRD_forecast_time_series.append(pd.DataFrame({"S": susceptible_population_forecast,
                                                                               "I": infected_population_forecast,
                                                                               "R": recovered_population_forecast,
                                                                               "D": dead_population_forecast}, index=[day]))

print(SIRD_forecast_time_series)

plt.plot(SIRD_forecast_time_series['I'].to_numpy())

plt.show()