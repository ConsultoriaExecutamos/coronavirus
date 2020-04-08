import datetime


starting_date = datetime.datetime(year=2020, month=2, day=27)

yesterday = datetime.datetime.now() - datetime.timedelta(days=1)

corrective_date_index = yesterday - starting_date

print(corrective_date_index.days)
