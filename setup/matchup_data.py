import pandas as pd
import logging
import csv
import os
from random import randint

def get_year_data(year, df):
    """
    Returns limited dataframe of just one year's matches
    :param year: int for year value
    :param df: pandas dataframe of full match data
    :return: abridged dataframe of one year's matches
    """
    year_data = df[df["Year"] == year]
    return year_data

def associate_data(match_data, comb_df):
    """
    Takes one match at a time and associates KenPom data with each team. Returns dictionary of associated
    data to be used for training
    :param bpi_df: dataframe of ESPN's BPI data
    :param match_data: one-row dataframe of match results
    :param kenpom_df: full kenpom dataframe for given year
    :return: Dict (final_data) with kenpom metrics and each team's score
    """

    score_dict = {"Team1": match_data["Team"],
                  "Score1": match_data["Score"],
                  "Team2": match_data["Team.1"],
                  "Score2": match_data["Score.1"],
                  "Year": match_data["Year"]}

    # logging.debug(f"Team1 is {score_dict['Team1']}, Team2 is {match_data['Team2']}")

    comb_df = comb_df[comb_df["Year"] == score_dict["Year"]]
    comb_df = comb_df[(comb_df["Team"] == match_data["Team"]) | (comb_df["Team"] == match_data["Team.1"])]

    # Want to randomize the team so the efficiency isn't always higher for team 1
    teamnum = randint(1,2)

    # final_data will hold kenpom & bpi data in dict of dicts
    final_data = {}
    # user iterrows() to get the team data as a dict
    for row in comb_df.iterrows():
        index, data = row
        # Extract team's data to a dictionary. Will use this to compile statistical data for each team
        data_dict = data.to_dict()

        # Add team's score in game to data_dict (logic to follow randomized team order)
        if data_dict["Team"] == score_dict["Team1"]:
            data_dict["Score"] = score_dict["Score1"]
        else:
            data_dict["Score"] = score_dict["Score2"]

        # Organize the kenpom data, append to final_dict
        final_data.update(organize_data(data_dict, teamnum))

        # Allow for random ordering of teams (This logic sets teamnum to the team that hasn't been added yet)
        if teamnum == 2:
            teamnum = 1
        else:
            teamnum = 2

    # Add column showing efficiency difference
    final_data["Eff_Diff"] = final_data["Eff1"] - final_data["Eff2"]

    # Add column showing score diff. This will be labeled result data
    final_data["Score_Diff"] = final_data['Score1'] - final_data['Score2']

    # determine game winner (0 or 1, for team1 or team2 respectively) and add to final_data
    # If score1 > score2, team1 wins. Else team 2 wins
    if final_data['Score1'] > final_data['Score2']:
        final_data['Winner'] = 0
    else:
        final_data['Winner'] = 1

    return final_data

def organize_data(data_dict, teamnum):
    """Organizes the important kenpom data into a dict
    This method is meant to be called from within associate_data().
    Creates the headers based on team number (1 or 2) so they can be inserted into csv"""

    # cast teamnum to string so it can be appended below
    teamnum = str(teamnum)
    output_dict = {}
    # compile all the important data into a dictionary here
    output_dict["Team" + teamnum] = data_dict["Team"]
    output_dict["Eff" + teamnum] = data_dict["AdjEM"]
    output_dict["Off" + teamnum] = data_dict["AdjO"]
    output_dict["Def" + teamnum] = data_dict["AdjD"]
    output_dict["SOS_Eff" + teamnum] = data_dict["Strength of Schedule AdjEM"]
    output_dict["SOS_Off" + teamnum] = data_dict["Strength of Schedule OppO"]
    output_dict["SOS_Def" + teamnum] = data_dict["Strength of Schedule OppD"]
    output_dict["Score" + teamnum] = data_dict["Score"]
    output_dict["BPI_Off" + teamnum] = data_dict["BPI Off"]
    output_dict["BPI_Def" + teamnum] = data_dict["BPI Def"]
    output_dict["BPI" + teamnum] = data_dict["BPI"]
    output_dict["Elo" + teamnum] = data_dict["Elo"]

    return output_dict

def data_to_csv(filename, training_data):

    logging.info(f"Appending matchup between {training_data['Team1']} and {training_data['Team2']}")
    fieldnames = ['Team1', 'Team2', 'Eff1', 'Eff2', "Off1", 'Off2', "Def1", 'Def2', 'SOS_Eff1', 'SOS_Eff2',
                  'SOS_Off1', 'SOS_Off2', 'SOS_Def1', 'SOS_Def2', "BPI_Off1", "BPI_Off2",
                  "BPI_Def1", "BPI_Def2", "BPI1", "BPI2", "Elo1", "Elo2", 'Score1', 'Score2', 'Eff_Diff', 'Score_Diff', 'Winner']
    write_headers = False

    # Check if the file exists already. If not, write headers to first row
    if not os.path.exists(filename):
        logging.debug("The file doesn't exist, write header row")
        write_headers = True

    # newline = '' comes from csv.dictwriter documentation
    with open(filename, "a", newline = '') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        writer.writerow(training_data)




