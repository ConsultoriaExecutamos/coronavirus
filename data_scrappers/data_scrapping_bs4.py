from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import threading


t0 = time.time()

main_worldometer_coronavirus_data_ulr = 'https://www.worldometers.info/coronavirus'

countries_worldometer_coronavirus_data_ulr = 'https://www.worldometers.info/coronavirus/#countries'

linked_countries_html = requests.get(countries_worldometer_coronavirus_data_ulr)

linked_countries_html_text = linked_countries_html.text

linked_countries_soup = BeautifulSoup(linked_countries_html_text, 'html.parser')

all_linked_countries_elements = linked_countries_soup.select("a[href*=country]")

all_linked_countries_list = []

for linked_country in all_linked_countries_elements[:int(len(all_linked_countries_elements))]:
    if linked_country.has_attr('href'):
        if linked_country['href'].split('/')[1] in all_linked_countries_list:
            break
        else:
            all_linked_countries_list.append(linked_country['href'].split('/')[1])


for country in all_linked_countries_list:

    country_worldometer_html = requests.get(main_worldometer_coronavirus_data_ulr + '/country/' + country)

    html_text = country_worldometer_html.text

    soup = BeautifulSoup(html_text, 'html.parser')

    index_list = [0, 0, 0, 0, 0]

    for i in range(len(soup.findAll("script"))):
        try:
            if 'coronavirus-cases-linear' in soup.findAll("script")[i].text.split('\n')[1]:
                index_list[0] = i
            elif 'graph-cases-daily' in soup.findAll("script")[i].text.split('\n')[1]:
                index_list[1] = i
            elif 'graph-active-cases-total' in soup.findAll("script")[i].text.split('\n')[1]:
                index_list[2] = i
            elif 'coronavirus-deaths-linear' in soup.findAll("script")[i].text.split('\n')[1]:
                index_list[3] = i
            elif 'graph-deaths-daily' in soup.findAll("script")[i].text.split('\n')[1]:
                index_list[4] = i
        except:
            pass


    total_cases_x_axys = soup.findAll("script")[index_list[0]].text.split('\n')[14].split(': ')[1].split('    ')[0][1:-1].replace('"', '').split(',')
    daily_new_cases_x_axys = soup.findAll("script")[index_list[1]].text.split('\n')[14].split(': ')[1].split('    ')[0][1:-1].replace('"', '').split(',')
    active_cases_x_axys = soup.findAll("script")[index_list[2]].text.split('\n')[14].split(': ')[1].split('    ')[0][1:-1].replace('"', '').split(',')
    total_deaths_x_axys = soup.findAll("script")[index_list[3]].text.split('\n')[14].split(': ')[1].split('    ')[0][1:-1].replace('"', '').split(',')
    daily_new_deaths_x_axys = soup.findAll("script")[index_list[4]].text.split('\n')[14].split(': ')[1].split('    ')[0][1:-1].replace('"', '').split(',')

    total_cases_y_axys = soup.findAll("script")[index_list[0]].text.split('\n')[38].split('[')[1].split(']')[0].split(',')
    daily_new_cases_y_axys = soup.findAll("script")[index_list[1]].text.split('\n')[36].split('[')[1].split(']')[0].split(',')
    active_cases_y_axys = soup.findAll("script")[index_list[2]].text.split('\n')[38].split('[')[1].split(']')[0].split(',')
    total_deaths_y_axys = soup.findAll("script")[index_list[3]].text.split('\n')[38].split('[')[1].split(']')[0].split(',')
    daily_new_deaths_y_axys = soup.findAll("script")[index_list[4]].text.split('\n')[36].split('[')[1].split(']')[0].split(',')

    data = []

    for i in range(len(total_cases_x_axys)):
        data.append([total_cases_x_axys[i], total_cases_y_axys[i], daily_new_cases_y_axys[i], active_cases_y_axys[i], total_deaths_y_axys[i], daily_new_deaths_y_axys[i]])

    dataframe = pd.DataFrame(data, columns=['Date', 'Total Cases', 'Daily New Cases', 'Active Cases', 'Total Deaths', 'Daily New Deaths'])

    dataframe.to_csv(country + '.csv', index=False)

    print(country)


print(time.time() - t0)

