import pandas as pd
from pprint import pprint

def get_xl_data():
    df_dict = pd.read_excel("KenPom_Rankings.xlsx", sheet_name = None, header = 0)
    pprint(df_dict["2002"])

def remove_unranked(df):
    # Todo: go through teams, check if they have a number next to them, if not set to NaN
    # Then drop the unranked teams from the dataframe
    ...

if __name__ == '__main__':
    get_xl_data()