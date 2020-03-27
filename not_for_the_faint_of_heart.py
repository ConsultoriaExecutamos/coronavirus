import json
import requests
import re
import os
import pandas as pd

from collections import OrderedDict
from pprint import pprint as pp
from lxml import html
from tqdm import tqdm
from timeit import default_timer as timer
from multiprocessing import Pool


URL = "https://www.worldometers.info/coronavirus/"

DAYS_PATTERN = "categories: \[(.*?)\]"
DATA_PATTERN = "data: \[(.*?)\]"

SELECTORS = OrderedDict()
SELECTORS["total"] = "div.graph_row:nth-child(1) > div:nth-child(1) > script:nth-child(4)"
SELECTORS["daily"] = "div.graph_row:nth-child(2) > div:nth-child(1) > script:nth-child(3)"
SELECTORS["active-total"] = "div.row:nth-child(3) > div:nth-child(1) > script:nth-child(3)"
SELECTORS["deaths-total"] = "div.row:nth-child(4) > div:nth-child(1) > script:nth-child(4)"
SELECTORS["deaths-daily"] = "div.row:nth-child(5) > div:nth-child(1) > script:nth-child(3)"


def country_to_csv(country):
    response = requests.get(URL + country)
    doc = html.fromstring(response.content)

    data = []

    for k, v in SELECTORS.items():
        chart_content = doc.cssselect(
            v)[0].text_content()
        if k == 'total':
            search = re.search(
                DAYS_PATTERN, chart_content, re.MULTILINE | re.DOTALL)
            data.append(search.group(1).split(","))
        search = re.search(
            DATA_PATTERN, chart_content, re.MULTILINE | re.DOTALL)
        data.append(search.group(1).split(","))

    dataframe = pd.DataFrame(data).T
    dataframe.columns = [
        'Date', 'Total Cases', 'Daily New Cases', 'Active Cases', 'Total Deaths', 'Daily New Deaths']

    dataframe.to_csv(f".\data\\{country[8:-1]}.csv", index=False)

    return country


def main():
    start = timer()

    response = requests.get(URL)
    doc = html.fromstring(response.content)
    countries = set()
    for element in doc.find_class("mt_a"):
        countries.add(element.attrib['href'])

    with Pool(16) as p:
        p.map(country_to_csv, countries)

    end = timer()
    print(f"Time elapsed: {end - start} seconds")


if __name__ == "__main__":
    main()
