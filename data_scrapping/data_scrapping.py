from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from datetime import datetime
from datetime import timedelta
import time
from selenium.webdriver.common.action_chains import ActionChains
import json
import openpyxl
import os
from selenium.webdriver.common.keys import Keys
import queue
import threading



########################################################################################################################
########################################################################################################################
########################################################################################################################


driver = webdriver.Chrome(executable_path='C:/Users/User/Downloads/chromedriver_win32/chromedriver.exe')

worldometer_coronavirus_info_site = 'https://www.worldometers.info/coronavirus/country/us'

driver.get(worldometer_coronavirus_info_site)

charts_list = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'graph_row')))

for chart in charts_list:
    chart_title = chart.find_elements_by_class_name('col-md-12')[0].text.split('\n')[0]
    script = chart.find_elements_by_tag_name('script')[0].get_attribute('innerHTML')
    print(script)



