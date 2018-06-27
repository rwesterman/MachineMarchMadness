import pandas as pd
import re
from pprint import pprint

def standardize_teams(match_df, df2, first_year):
    """
    Alters team names in Match_Data.csv to correspond to KenPom team names
    :param match_df: dataframe from Match_Data.csv
    :param df2: dataframe from KenPom_Complete.csv
    :param first_year: First year of data for df2
    :return: match_updated dataframe
    """

    match_df = match_df[match_df["Year"] >= first_year]

    # team1 and team2 are Series, respectively columns Team and Team.1
    team1 = match_df.loc[:,"Team"]
    team2 = match_df.loc[:,"Team.1"]

    # Get "Team" column from kenpom dataframe
    df2_team = df2.loc[:, "Team"]

    # Change all teams that end with St to St. in Match Data
    df2_team = df2_team.apply(normalize_st)
    df2_team = df2_team.apply(normalize_state)

    find_differences(team1, team2, df2_team)

    df2_team = df2_team.apply(change_outliers)

    df2["Team"] = df2_team

    find_differences(team1, team2, df2_team)

    return df2

    # match_df.to_csv("Training_Data\\Match_Data.csv", index = False)


def normalize_st(team):
    """
    normalizes team names between Match_Data.csv and KenPom_Complete.csv
    :param team: name of team, run regex to check for certain attributes
    :return: modified or unmodified name of team
    """
    # $ symbol means that it only looks at the end of the string
    return re.sub(r'St$', 'St.', team)

def normalize_state(team):
    return re.sub(r'State', 'St.', team)

def change_outliers(team):
    outliers = {"Miami": "Miami FL",
                "Long Island Brooklyn": "LIU Brooklyn",
                "Central Connecticut St.": "Central Connecticut",
                "St Josephs": "Saint Joseph's",
                "St Marys": "Saint Mary's",
                "St Peters": "Saint Peter's",
                "St Johns": "St. John's",
                "St Bonaventure": "St. Bonaventure",
                "Southern Mississippi":"Southern Miss",
                "Stephen F Austin": "Stephen F. Austin",
                "Texas A&M Corpus Christi":"Texas A&M Corpus Chris",
                "Cal Irvine": "UC Irvine",
                "Santa Barbara":"UC Santa Barbara",
                "Central Florida":"UCF",
                "Texas Arlington":"UT Arlington",
                "Texas San Antonio":"UTSA",
                "Pennsylvania": "Penn",
                "Miami Ohio": "Miami OH",
                "Loyola Maryland": "Loyola MD",
                "Middle Tennessee State":"Middle Tennessee",
                "Mount St Marys": "Mount St. Mary's",
                "NC State":"North Carolina St.",
                "St Louis": "Saint Louis",
                "Ole Miss": "Mississippi",
                "Wisconsin Milwaukee":"Milwaukee",
                "Middle Tennessee St.":"Middle Tennessee",
                "Wisconsin Green Bay": "Green Bay",
                "UT-Arlington": "UT Arlington",
                "UT San Antonio": "UTSA",
                "Loyola (MD)": "Loyola MD",
                "Loyola (Chi)": "Loyola Chicago",
                "Mt. St. Mary's" : "Mount St. Mary's",
                "N Colorado" : "Northern Colorado",
                "Ohio State" : "Ohio St.",
                "Charleston" : "College of Charleston",
                "CSU Northridge": "Cal St. Northridge",
                "UConn": "Connecticut",
                "CSU Fullerton": "Cal St. Fullerton",
                "CSU Bakersfield": "Cal St. Bakersfield",
                "Boston Univ." : "Boston University",
                "Miss Valley St.": "Mississippi Valley St.",
                "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
                "Arkansas-Little Rock": "Arkansas Little Rock",
                "Detroit Mercy" : "Detroit",
                "Hawai'i": "Hawaii",
                "Brigham Young": "BYU",
                "UC-Irvine": "UC Irvine",
                "UC-Davis": "UC Davis",
                "UC-Santa Barbara": "UC Santa Barbara",
                "Texas Christian": "TCU",
                "Alabama-Birmingham": "UAB",
                "Maryland-Baltimore County": "UMBC",
                "North Carolina-Asheville": "UNC Asheville",
                "North Carolina-Greensboro": "UNC Greensboro",
                "North Carolina-Wilmington": "UNC Wilmington",
                "Albany (NY)": "Albany",
                "Little Rock": "Arkansas Little Rock",
                "University of California": "California",
                "Louisiana State": "LSU",
                "Louisiana St.": "LSU",
                "Long Island University": "LIU Brooklyn",
                "Loyola (IL)": "Loyola Chicago",
                "Miami (FL)": "Miami FL",
                "Miami (OH)": "Miami OH",
                "Southern Methodist": "SMU",
                "St. John's (NY)": "St. John's",
                "Saint Mary's (CA)": "Saint Mary's",
                "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
                "Nevada-Las Vegas": "UNLV",
                "Southern California": "USC",
                "Texas-Arlington": "UT Arlington",
                "Texas-El Paso": "UTEP",
                "Texas-San Antonio": "UTSA",
                "Virginia Commonwealth": "VCU",
                "Illinois-Chicago": "Illinois Chicago"
                }

    # If parameter team is an outlier, return the replacement function
    if team in outliers:
        return outliers[team]
    else:
        return team

def find_differences(m_team1, m_team2, df2_team):
    """
    Standardizing to match_data.csv, so finding all differences between teams in tournament and teams in other datasets
    :param m_team1: match_data team1 name Series
    :param m_team2: match_data team2 name Series
    :param df2_team: kenpom team name Series
    :return:
    """
    match_set = set()
    df2_set = set()

    match_set = match_set.union(set(m_team1.tolist()))
    match_set = match_set.union(set(m_team2.tolist()))

    df2_set = df2_set.union(set(df2_team.tolist()))
    # #
    outliers = match_set - df2_set
    #
    pprint(outliers)
    return outliers