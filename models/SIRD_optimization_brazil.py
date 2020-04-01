import pandas as pd
import os
import requests
from io import StringIO
import datetime


today = datetime.datetime.now()

print(today)

brazil_file_url = 'https://covid.saude.gov.br/assets/files/COVID19_20200401.csv'

r = requests.get(brazil_file_url).text

brazil_states_dataframe = pd.read_csv(StringIO(r), delimiter=';')
#
#
#
# rj_time_series = brazil_states_dataframe.loc[brazil_states_dataframe['estado'] == 'RJ']
#
# total_cases = rj_time_series['casosAcumulados']
#
# print(total_cases)