from setup import matchup_data, clean_tables, standardize_data
import logging
import os
import pandas as pd
from setup.data_classes import Data_Frames

def normalize_training_set(filepath):
    train_data = pd.read_csv(filepath)

    print(train_data.head())

    ignore_columns = ["Winner", "Score1", "Score2", "Score_Diff"]

    for column in train_data:
        # Get max and min values from the column in question
        max = train_data[column].max()
        min = train_data[column].min()

        if type(max) == str:
            # If on a column of string values, move to next column
            continue

        elif column in ignore_columns:
            # Ignore non-training columns
            continue

        # normalize all data in the column
        train_data[column] = train_data[column].apply(normalize_data, args = (max, min))


    train_data.to_csv(".\\Training_Data\\Training_Set_Normalized.csv", index = False)


def normalize_data(value, max, min):
    """Normalizes every value in a column between [-1,1]"""

    # Get max and min values in the column

    try:
        max = float(max)
        min = float(min)

    except ValueError as e:
        logging.warning("Cannot normalize strings")

    # formula for normalizing data between -1 and 1
    # found at: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    return 2*((value - min)/(max - min)) - 1




def compile_data(start_year, end_year, result_path, comb_df, match_data):

    # Add 1 to end_year because range is not inclusive
    for year in range(start_year, end_year + 1):

        # get_year_data reduces dataframe to only data from a single year.
        year_data = matchup_data.get_year_data(year, match_data)

        # could just do itertuples here instead?
        for x in range(len(year_data.index)):
            single_match = year_data.iloc[x]

            # Associate the data from a single match (by matching those teams to the comp_df teams)
            try:
                kp_data = matchup_data.associate_data(single_match, comb_df)
                matchup_data.data_to_csv(result_path, kp_data)

            # Catch exceptions if a team can't be added
            except KeyError as e:
                print(f"Error adding {single_match['Team']} or "
                      f"{single_match['Team.1']} from year {single_match['Year']}")
                print(f"KeyError: {e}")

def prepare_compile():
    kp_path = ".\\Training_Data\\KenPom_Complete.csv"
    bpi_path = ".\\Training_Data\\ESPN_BPI.csv"
    elo_path = ".\\Training_Data\\elo_rankings.csv"
    match_path = ".\\Training_Data\\Match_Data.csv"

    data = Data_Frames(kp_path, bpi_path, elo_path, match_path)
    elo_df = data.get_elo_df()
    bpi_df = data.get_bpi_df()
    kp_df = data.get_kp_df()
    match_df = data.get_match_df()

    comp_df = data.combine_dfs(match_df, kp_df, bpi_df, elo_df)
    # elo_df = standardize_data.standardize_teams(match_df, elo_df, 2003)

    # elo_df.to_csv(elo_path, index = False)
    #
    # if os.path.exists(".\\Training_Data\\Training_Set.csv"):
    #     os.remove(".\\Training_Data\\Training_Set.csv")
    compile_data(2008, 2018, ".\\Training_Data\\Training_Set.csv", comp_df, match_df)
    # match_df = matchup_data.get_match_data()
    # bpi_df = matchup_data.get_bpi_df()

    # bpi_df = standardize_data.standardize_teams(match_df, bpi_df, 2008)
    # bpi_df.to_csv(".\\Training_Data\\ESPN_BPI.csv", index = False)

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    # Delete existing training set if it exists

    normalize_training_set(".\\Training_Data\\Training_Set.csv")
    # prepare_compile()

