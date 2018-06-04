import pandas as pd
import re
from pprint import pprint

def get_xl_data():
    df_dict = pd.read_excel("KenPom_Rankings.xlsx", sheet_name = None, header = 0)
    return df_dict

def remove_unranked(df):
    # Add a boolean column checking if a team is ranked (has a number next to team name)
    df["ranked?"] = df.Team.str.contains('[0-9]+')

    # Create new dataframe of only teams that are ranked
    new_df = df[df["ranked?"] == True]

    # drop now-unnecessary "ranked?" column from new dataframe
    new_df.drop("ranked?", axis = 1, inplace = True)

    return new_df

def extract_mm_rank(df):

    # Split only the last space between team name and seed number
    teamrank = df.Team.str.rsplit(" ", 1)

    # Get the element 0 (team name) and element 1 (seed) separately
    team = teamrank.str.get(0)
    seed = teamrank.str.get(1)

    # Replace original Team column with new column of just team name, and create Seed column
    df["Team"] = team
    df["Seed"] = seed

    return df

def df_to_csv(year, df):
    # output new dataframe as .csv
    new_df.to_csv(f"KenPom_{year}.csv", index = False)

if __name__ == '__main__':
    # Ignore warning about using .loc to replace values in dataframe
    pd.options.mode.chained_assignment = None  # default='warn'

    df_dict = get_xl_data()

    for year, df in df_dict.items():
        # Remove the unranked teams from the data
        new_df = remove_unranked(df)
        # Extract the ncaa tournament seed from the team name
        new_df = extract_mm_rank(new_df)
        # Output the result to csv
        df_to_csv(year, df)
        # replace the dataframe in df_dict with the altered dataframe
        df_dict[year] = new_df

