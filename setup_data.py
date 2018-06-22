from setup import matchup_data, clean_tables, standardize_data
import logging
import os
import pandas as pd

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

    #
    # if os.path.exists(".\\Training_Data\\Training_Set.csv"):
    #     os.remove(".\\Training_Data\\Training_Set.csv")
    compile_data(2008, 2018, ".\\Training_Data\\Training_Set.csv")
    # match_df = matchup_data.get_match_data()
    # bpi_df = matchup_data.get_bpi_df()
    # bpi_df = standardize_data.standardize_teams(match_df, bpi_df, 2008)
    # bpi_df.to_csv(".\\Training_Data\\ESPN_BPI.csv", index = False)