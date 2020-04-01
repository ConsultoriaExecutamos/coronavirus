import pandas as pd
import requests
from io import StringIO
import datetime
import os


root_dir = os.path.join(os.path.dirname( __file__ ), '..' )

today = datetime.datetime.now().strftime('%Y%m%d')

brazil_file_url = 'https://covid.saude.gov.br/assets/files/COVID19_' + today + '.csv'

r = requests.get(brazil_file_url).text

brazil_states_dataframe = pd.read_csv(StringIO(r), delimiter=';')

brazil_states_dataframe.to_csv(root_dir + '/data/brazillian_states_data/COVID19_' + today + '.csv', index=False)

