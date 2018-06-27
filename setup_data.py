from setup import matchup_data, clean_tables, standardize_data
import logging
import os
import pandas as pd
from setup.data_classes import Data_Frames
def compile_data(start_year, end_year, result_path):
    df = matchup_data.get_match_data()
    kp_comp = matchup_data.get_kenpom_df()
    bpi_df = matchup_data.get_bpi_df()

    # Add 1 to end_year because range is not inclusive
    for year in range(start_year, end_year + 1):
        year_data = matchup_data.get_year_data(year, df)
        for x in range(len(year_data.index)):
            single_match = year_data.iloc[x]

            try:
                kp_data = matchup_data.associate_data(single_match, kp_comp, bpi_df)
                matchup_data.data_to_csv(result_path, kp_data)
            except KeyError as e:
                print(f"Error adding {single_match['Team']} or "
                      f"{single_match['Team.1']} from year {single_match['Year']}")
                print(f"KeyError: {e}")


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    # Delete existing training set if it exists


    kp_path = ".\\Training_Data\\KenPom_Complete.csv"
    bpi_path = ".\\Training_Data\\ESPN_BPI.csv"
    elo_path = ".\\Training_Data\\elo_rankings.csv"
    match_path = ".\\Training_Data\\Match_Data.csv"

    data = Data_Frames(kp_path, bpi_path, elo_path, match_path)
    elo_df = data.get_elo_df()
    bpi_df = data.get_bpi_df()
    kp_df = data.get_kp_df()
    match_df = data.get_match_df()

    elo_df = standardize_data.standardize_teams(match_df, elo_df, 2003)
    print(elo_df.head(20))
    elo_df.to_csv(elo_path, index = False)
    #
    # if os.path.exists(".\\Training_Data\\Training_Set.csv"):
    #     os.remove(".\\Training_Data\\Training_Set.csv")
    # compile_data(2008, 2018, ".\\Training_Data\\Training_Set.csv")
    # match_df = matchup_data.get_match_data()
    # bpi_df = matchup_data.get_bpi_df()

    # bpi_df = standardize_data.standardize_teams(match_df, bpi_df, 2008)
    # bpi_df.to_csv(".\\Training_Data\\ESPN_BPI.csv", index = False)