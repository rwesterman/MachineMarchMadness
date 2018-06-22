import pandas as pd
import re
from pprint import pprint

def get_xl_data():
    """
    Returns a dictionary of dataframes where each key is a sheet name and each value is a dataframe
    :return:
    """
    # Setting sheet_name = None imports all sheets
    df_dict = pd.read_excel("Training_Data\\KenPom_Rankings.xlsx", sheet_name = None, header = 0)

    return df_dict

def remove_unranked(df):
    # Add a boolean column checking if a team is ranked (has a number next to team name)
    df["ranked?"] = df.Team.str.contains('[0-9]+')

    # Create new dataframe of only teams that are ranked
    new_df = df[df["ranked?"] == True]

    # drop now-unnecessary "ranked?" column from new dataframe
    new_df = new_df.drop("ranked?", axis = 1)
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

def df_to_csv(title, df):
    # output new dataframe as .csv
    df.to_csv(f"Training_Data\\KenPom_{title}.csv", index = False)

def add_year_col(year, df):
    df["Year"] = year

    return df


def set_kenpom_complete():
    kenpom_complete = pd.DataFrame()
    df_dict = get_xl_data()

    for year, df in df_dict.items():
        # Remove the unranked teams from the data
        new_df = remove_unranked(df)
        # # Extract the ncaa tournament seed from the team name
        new_df = extract_mm_rank(new_df)
        # # Add a column for the current year
        new_df = add_year_col(year, new_df)
        # Add the yearly dataframe to a compilation df of all years
        kenpom_complete = kenpom_complete.append(new_df)

    # Output the result to csv
    df_to_csv("Complete", kenpom_complete)
    return kenpom_complete


if __name__ == '__main__':

    # Ignore warning about using .loc to replace values in dataframe
    pd.options.mode.chained_assignment = None  # default='warn'

    # Get kenpom complete dataframe
    kp_comp = set_kenpom_complete()
