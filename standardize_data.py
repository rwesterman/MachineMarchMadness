import pandas as pd
import re
from matchup_data import get_match_data, import_kenpom_df
from pprint import pprint

def standardize_teams(match_df, kp_df):
    """
    Alters team names in Match_Data.csv to correspond to KenPom team names
    :param match_df: dataframe from Match_Data.csv
    :param kp_df: dataframe from KenPom_Complete.csv
    :return: match_updated dataframe
    """
    # team1 and team2 are Series, respectively columns Team and Team.1
    team1 = match_df.loc[:,"Team"]
    team2 = match_df.loc[:,"Team.1"]

    kp_team = kp_df.loc[:,"Team"]

    find_differences(team1, team2, kp_team)

    # Change all teams that end with St to St. in Match Data
    team1 = team1.apply(normalize_state)
    team2 = team2.apply(normalize_state)


def normalize_state(team):
    """
    normalizes team names between Match_Data.csv and KenPom_Complete.csv
    :param team: name of team, run regex to check for certain attributes
    :return: modified or unmodified name of team
    """
    return re.sub(r'St$', 'St.', team)

def change_outliers(team):
    outliers = {"Miami": "Miami FL",
                "Long Island Brooklyn": "LIU Brooklyn",
                "Central Connecticut St.": "Central Connecticut",
                }

def find_differences(m_team1, m_team2, kp_team):
    """

    :param m_team1: match_data team1 name Series
    :param m_team2: match_data team2 name Series
    :param kp_team: kenpom team name Series
    :return:
    """
    match_set = set()
    kp_set = set()

    match_set = match_set.union(set(m_team1.tolist()))
    match_set = match_set.union(set(m_team2.tolist()))

    kp_set = kp_set.union(set(kp_team.tolist()))
    # #
    outliers = kp_set - match_set
    #
    pprint(outliers)



if __name__ == '__main__':
    standardize_teams(get_match_data(), import_kenpom_df())