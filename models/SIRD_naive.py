import matplotlib.pyplot as plt
import pandas as pd
import os
import openpyxl

########################################################################################################################
########################################################################################################################
########################################################################################################################
# load country data
root_dir = os.path.join(os.path.dirname( __file__ ), '..' )

time_series_df = pd.read_csv(root_dir + '/data/countries_time_series/brazil.csv')

total_cases_time_series = time_series_df['Total Cases'].to_numpy()

active_cases_time_series = time_series_df['Active Cases'].to_numpy()

total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()

recover_cases_time_series = pd.DataFrame(total_cases_time_series - active_cases_time_series - total_deaths_time_series)

time_series_df.insert(len(time_series_df.columns), 'Recover Cases', recover_cases_time_series)

demographics_df = pd.read_csv(root_dir + '/data/countries_demographics/countries_demographics.csv')

country_population = demographics_df['Population'].values[demographics_df.index[demographics_df['Country'] == 'Brazil'].tolist()[0]]



# initial model values
S = 1000                # susceptible population
I = 15                  # infected population
R = 0                   # recovered population
D = 0                   # dead population

forecast_steps = 50

# disease params
alpha = .02         # infection rate
beta = .0005        # recovery rate
gamma = .05         # fatality rate


# history = pd.DataFrame({"S": S, "I": I, "R": R, "D": D}, index=[0])
#
# #Run sim loop
#
# history["step"] = history.index
# plotData = pd.melt(history, id_vars=["step"])
# ggplot(plotData, aes(x="step", y="value", color="variable"))+geom_line()
# for step in range(1, steps):
#     newInf = floor(min(max(beta*I*S, 0), S))
#     newRec = floor(min(max(gamma*I, 0), I))
#     newDead = floor(min(max(mu*I, 0), I-newRec))
#     S = S - newInf
#     I = I + newInf - newRec - newDead
#     R = R + newRec
#     D = D + newDead
#     history = history.append(pd.DataFrame({"S": S, "I": I, "R": R, "D": D}, index=[step]))
# history["step"] = history.index
# #Plot using Python port of ggplot
# plotData = pd.melt(history, id_vars=["step"], value_vars=["S","I","R","D"])
# ggplot(plotData, aes(x="step", y="value", color="variable"))+geom_line()+xlab("Time Step")+ylab("# Hosts")