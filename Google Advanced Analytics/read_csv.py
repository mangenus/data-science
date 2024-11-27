""""
with open('./Unicorn_Companies.csv', mode='r') as file:
    data = file.read()
"""

import pandas as pd

df = pd.read_csv('./Unicorn_Companies.csv')
