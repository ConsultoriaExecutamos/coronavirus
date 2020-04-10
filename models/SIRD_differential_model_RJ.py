import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import differential_evolution
import datetime
import json


def SIRD_model(y, t, N, X):
    # SIRD model differential equations.
    S, IN, IA, IS, R, D, SUM = y
    dSdt = (-X[0]*S*(IA/N))
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

    time_series_columns_list = ['Total Deaths']
    forecasts_list = [D]

    sum_of_errors = 0

    for i in range(len(forecasts_list)):
        # Crop time series to retrieve data with at least one infected.
        actual_vector = time_series_df[time_series_columns_list[i]].values[df_index:]
        forecasted_vector = forecasts_list[i]
        actual_vector+=1    # In order to keep the operation viable when actual vector has zeros.
        errors = np.abs((forecasted_vector/actual_vector)-1)
        sum_of_errors += np.sum(errors)

    return sum_of_errors


def obtain_best_fit_estimators(y0, N, t, bounds):
    optimized_SIRD_model_result = differential_evolution(SIRD_model_fitting, bounds=bounds, args=(y0, N, t,),
                                                         maxiter=50, disp=True, polish=True, mutation=(0,1.9),workers=1,
                                                         seed=1)
    return optimized_SIRD_model_result.x


def plot_SIRD_model(S, IN, IA, IS, R, D, SUM,t):

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, IN, 'k', alpha=0.5, lw=2, label='Infected Incubation - Forecast')
    ax.plot(t, IA, 'c', alpha=0.5, lw=2, label='Infected Active - Forecast')
    ax.plot(t, IS, 'm', alpha=0.5, lw=2, label='Infected Isolated - Forecast')
    ax.plot(t, IN + IA + IS, 'r', alpha=0.5, lw=2, label='Infected Total - Forecast')
    ax.plot(t, SUM, 'b', alpha=0.5, lw=2, label='Total Cases - Forecast')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity - Forecast')
    ax.plot(t, D, 'y', alpha=0.5, lw=2, label='Dead - Forecast')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')

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

    # Load and set time series data.
    time_series_df = pd.read_csv(root_dir + '/data/brazillian_states_time_series/RJ_updated.csv')
    total_cases_time_series = time_series_df['Total Cases'].to_numpy()
    active_cases_time_series = time_series_df['Active Cases'].to_numpy()
    total_deaths_time_series = time_series_df['Total Deaths'].to_numpy()


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

    df_index -= 1

    current_infected_population = time_series_df['Active Cases'].values[df_index]
    recovered_population = time_series_df['Total Recovery Cases'].values[df_index]
    dead_population = time_series_df['Total Deaths'].values[df_index]

    pd.set_option('display.max_columns', None)

    S0 = N = 2200/0.019    # Gompertz Model output - 3k and the average of the mortality rate bounds (1,4% and 2,5%).

    y0 = S0, current_infected_population, recovered_population, 0, 0, 0, 0

    # Determine time grid for fitting.
    t_length = len(time_series_df) - df_index
    t = t = np.linspace(0, t_length, t_length)

    # Allow it to be as free as possible, as long as it is acceptable.
    alpha_bounds = (0, 0.6)
    # The recovery parameter. We allowed it to bounce between 4 (1/0.25) days and 10 (1/0.1) days, considering the sum
    # on incubation and isolation time.
    beta_bounds = (0.1, 0.25)
    # The median time of death since incubation date is 20 days, according to Wang et.al (2020) - Estimating clinical
    # severity of COVID-19 from the transmission dynamics in Wuhan, China. Also, we considered that the lower bound for
    # fatality rate is 0.014 and the upper bound is 0.025. We allowed it to bounce between 5 (0.0009/0.014) days and 15
    # (0.005/0.025) days, considering the sum on incubation and isolation time.
    gamma_bounds = (0.0009, 0.005)
    # The median time of incubation is 5 days, according to Wang et.al (2020) - Estimating clinical severity of
    # COVID-19 from the transmission dynamics in Wuhan, China. We allowed it to bounce between 3.3 (1/0.3)
    # days and 7.14 (1/0.14) days.
    tetha_bound = (0.14, 0.3)
    # Isolation parameter. We allowed it to bounce between 5 (1/0.2) days and 14 (1/0.0588) days.
    delta_bound = (0.0588, 0.2)

    bounds = [alpha_bounds, beta_bounds, gamma_bounds, tetha_bound, delta_bound]

    best_fit_estimators = obtain_best_fit_estimators(y0, N, t, bounds)

    # Determine time grid for projections.
    t = t = np.linspace(0, 120, 120)
    print(best_fit_estimators)

    S, IN, IA, IS, R, D, SUM = SIRD_model_sim(y0, t, N, best_fit_estimators)
    plot_SIRD_model(S, IN, IA, IS, R, D, SUM,t)

    beggining_of_ajusted_time_series = datetime.datetime(year=2020, month=2, day=27)

    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)

    corrective_date_index = (yesterday - beggining_of_ajusted_time_series).days

    list_S = S.tolist()[corrective_date_index:]
    list_IN = IN.tolist()[corrective_date_index:]
    list_IA = IA.tolist()[corrective_date_index:]
    list_IS = IS.tolist()[corrective_date_index:]
    list_R = R.tolist()[corrective_date_index:]
    list_D = D.tolist()[corrective_date_index:]
    list_SUM = SUM.tolist()[corrective_date_index:]
    list_TI = IN[corrective_date_index:] + IA[corrective_date_index:] + IS[corrective_date_index:]

    S_data = []
    IN_data = []
    IA_data = []
    IS_data = []
    R_data = []
    D_data = []
    SUM_data = []
    TI_data = []

    all_lists_list = [list_S, list_IN, list_IA, list_IS, list_R, list_D, list_SUM, list_TI]

    all_data_list = [S_data, IN_data, IA_data, IS_data, R_data, D_data, SUM_data, TI_data]

    time_delta = datetime.timedelta(days=1)

    for i in range(len(all_data_list)):
        for j in range(len(list_S)):
            all_data_list[i].append([int(datetime.datetime.timestamp(yesterday + time_delta * j)) * 1000, int(all_lists_list[i][j])])

    print(all_data_list[0])
    print(all_data_list[1])
    print(all_data_list[2])
    print(all_data_list[3])
    print(all_data_list[4])
    print(all_data_list[5])
    print(all_data_list[6])
    print(all_data_list[7])

    with open(root_dir + '/data/brazillian_states_time_series/RJ.json', 'w') as outfile:
        json.dump(all_data_list, outfile)

