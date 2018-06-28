import pandas as pd
import math
from collections import defaultdict
import test_model
import training
from torch.autograd import Variable
import torch
import logging

# Todo: Add scoring system comparing predicted bracket to actual bracket

# Todo: Make eventual output in format: (team1 name, seed) vs (team2 name, seed) -> (winner name, seed)

# Todo: Need to figure out how to determine next opponent based on victory. This might work recursively??

# Note: Year selection occurs at the kenpom_df() level. Everything else follows from that year

def extrapolate_seed(seed, region):
    """
    Return the 64_bracket seed value for a given seed and region
    :param seed: int, 16 team seed
    :param region: int, region number (1-4)
    :return: 64_bracket seed value
    """
    # to extraopolate to 64 team seed, take current seed and add (16 * (region number - 1))
    # This works because regions are numbered 1 through 4, so region 1 keeps its seed numbers, each other region is 16 total seeds later
    new_seed = seed + (16 * (region - 1))
    return new_seed

def get_matchup_data(team1, team2, df):
    """
    :param team1: Name of team1 in matchup
    :param team2: Name of team2 in matchup
    :param df: kenpom dataframe, passing through to get_team_data()
    :return: a torch Tensor of stats for two teams
    """
    # Todo: Might want to randomize the order here, since that's how the model was trained

    team1_data = get_team_data(df, team1)
    team2_data = get_team_data(df, team2)

    # Todo: Create large dict here with values for each team

    # Building a dictionary that has the same columns as the training data

    stats = []

    stats.append(team1_data.iloc[0]["AdjEM"])
    stats.append(team2_data.iloc[0]["AdjEM"])

    stats.append(team1_data.iloc[0]["AdjO"])
    stats.append(team2_data.iloc[0]["AdjO"])

    stats.append(team1_data.iloc[0]["AdjD"])
    stats.append(team2_data.iloc[0]["AdjD"])

    stats.append(team1_data.iloc[0]["Strength of Schedule AdjEM"])
    stats.append(team2_data.iloc[0]["Strength of Schedule AdjEM"])

    stats.append(team1_data.iloc[0]["Strength of Schedule OppO"])
    stats.append(team2_data.iloc[0]["Strength of Schedule OppO"])

    stats.append(team1_data.iloc[0]["Strength of Schedule OppD"])
    stats.append(team2_data.iloc[0]["Strength of Schedule OppD"])

    stats = torch.FloatTensor(stats)
    return stats

def get_team_data(df, team_name):
    """

    :param df: Takes in KenPom dataframe
    :param team_name: Name of team to get data
    :return: return pandas dataframe of team's KenPom data
    """

    df = df[df["Team"] == team_name]

    # Should limit to one row dataframe (excluding header row)
    return df


def match_teams_seeds(match_path, year):
    """
    :param match_path: filepath for match_data.csv
    :param year: year of tournament
    :return: tourney_dict is a dictionary where each key is 64_bracket seed number, and each value is the team's name
    """
    df = pd.read_csv(match_path, index_col="Year")
    df = df.loc[year]

    # All tournament teams are listed in Round 1
    df = df[df["Round"] == 1]

    tourney_dict = {}

    # Iterate through every row of the Match_Data.csv dataframe
    for row in df.itertuples():

        # row indexing - 1: Round, 2: Region Number, 3: Region Name, 4: Seed, 5: Score,
        # 6: Team, 7: Team.1, 8: Score.1, 9: Seed.1

        # to get 64_bracket version of team seed, take original seed (row[4]) and add 16 * (region # - 1)
        team1_seed = extrapolate_seed(row[4], row[2])
        team2_seed = extrapolate_seed(row[9], row[2])

        team1_name = row[6]
        team2_name = row[7]

        tourney_dict[team1_seed] = team1_name
        tourney_dict[team2_seed] = team2_name

    logging.debug("Tourney_dict is {}".format(tourney_dict))
    return tourney_dict

# This is an incredible little algorithm
# https://codereview.stackexchange.com/questions/17703/using-python-to-model-a-single-elimination-tournament
def tournament_round(num_teams, matchlist):
    """
    Recursively builds list of lists, each time it is called. Final output will be list of lists log2(num_teams) + 1 deep
    :param num_teams:
    :param matchlist:
    :return:
    """
    new_matches = []

    for team_or_match in matchlist:
        # Checks if team_or_match is of type list. since this function creates list of lists on first pass,
        # second pass this for loop gives team_or_match type == list
        if type(team_or_match) == type([]):
            new_matches += [tournament_round(num_teams, team_or_match)]
        else:
            new_matches += [[team_or_match, num_teams + 1 - team_or_match]]
    return new_matches

def flatten_list(matches):
    """Takes a list of lists, log2(num_teams)+1 lists deep, then flattens them recursively to single level"""
    teamlist = []
    for team_or_match in matches:
        if type(team_or_match) == type([]):
            teamlist += flatten_list(team_or_match)
        else:
            teamlist += [team_or_match]
    return teamlist

def generate_tournament(num_teams):
    """Generates a tournament?"""

    # The number of rounds can be calculated as log base 2 of number of teams
    num_rounds = math.log(num_teams, 2)

    # Catch exceptions if log2(num_teams) is not an integer
    if num_rounds != math.trunc( num_rounds ):
        raise ValueError("Number of teams must be a power of 2")

    # Initialize teams and result
    # teams will act as input to track
    teams = 1
    result = [1]
    while teams != num_teams:
        teams *= 2
        result = tournament_round(teams, result)
    return flatten_list(result)

def kenpom_df(filepath, year):
    """
    Returns pandas dataframe for one year of KenPom rankings
    :param filepath: path to KenPom_Complete.csv
    :return: KenPom dataframe
    """

    df = pd.read_csv(filepath)
    df = df[df["Year"] == year]
    return df

def seed_from_tourneydict(tourney_dict, seed, team_name):
    """
    Compare team name against tourney_dict to get region-adjusted seed
    :param tourney_dict: Dictionary of key:value pairs in form seed:team_name
    :param starting_seed: Seed not yet adjusted for region
    :param team_name: Name of team to search for
    :return:
    """
    # Since regions don't apply to championship, need to find region adjusted seed of winner,
    # do this by adding 16 to original seed until the value in tourney_dict matches the team name
    iterate_seed = seed
    while iterate_seed <= 64:
        if tourney_dict[iterate_seed] == team_name:
            seed = iterate_seed
        iterate_seed += 16
    return seed


def actual_bracket(year, round, tourney_dict):
    """
    Dynmically pulls actual winners for a given year and round from match_data.csv
    :param year: year of tournament (int)
    :param round: round of tournament (int)
    :param tourney_dict: Dictionary of seed: team_name, used here to check correct seed for teams in rounds 5 and 6
    :return: list of winners for given year and round
    """

    winners = []

    # region numbers are kept consistent with those shown in match_data.csv, which means they
    # change between years. Make sure these are kept consistent across data.

    # Order is unimportant in these matchups because the scoring comparison simply checks if a value is in the list

    df = pd.read_csv(".\\Training_Data\\Match_Data.csv", index_col = "Year")
    # limit df to requested year
    df = df.loc[year]
    # further limit df to requested round + 1 (Plus one so we can infer the winners of the requested round)

    if round < 5:
        df = df[df["Round"] == round + 1]
        logging.debug(df.head())

        # add the seed of each team in the round following the requested round, because it means they won the requested round
        for row in df.itertuples():
            # extrapolate the region based seed from seed number and region number, add to winners list
            winners.append(seed_from_tourneydict(tourney_dict, row[4], row[6]))
            winners.append(seed_from_tourneydict(tourney_dict, row[9], row[7]))
            # winners.append(extrapolate_seed(row[4], row[2]))
            # winners.append(extrapolate_seed(row[9], row[2]))

    elif round == 5:
        # This is the Final Four round, so seeds can't be extrapolated from region numbers
        # Therefore have to do it by checking against tourney_dict
        df = df[df["Round"] == 6]
        # Go through each team the final four, get their seed number from tourney_dict, add to winners list
        team1_seed = df.iloc[0]["Seed"]
        team1_name = df.iloc[0]["Team"]
        team2_name = df.iloc[0]["Team.1"]
        team2_seed = df.iloc[0]["Seed.1"]

        team1_seed = seed_from_tourneydict(tourney_dict, team1_seed, team1_name)
        team2_seed = seed_from_tourneydict(tourney_dict, team2_seed, team2_name)

        winners.append(team1_seed)
        winners.append(team2_seed)


    # If looking for results of the final round, need to compare scores, then backtrack correct seed number
    else:
        df = df[df["Round"] == 6]
        # Compare scores in final game to determine winner
        if df.iloc[0]["Score"] > df.iloc[0]["Score.1"]:
            win_team_name = df.iloc[0]["Team"]
            win_team_seed = df.iloc[0]["Seed"]
        else:
            win_team_name = df.iloc[0]["Team.1"]
            win_team_seed = df.iloc[0]["Seed.1"]

        # Since regions don't apply to championship, need to find region adjusted seed of winner,
        # do this by adding 16 to original seed until the value in tourney_dict matches the team name
        win_team_seed = seed_from_tourneydict(tourney_dict, win_team_seed, win_team_name)
        winners.append(win_team_seed)
    return winners

def score_per_round(round, pred_winner, actual_winner):
    # number of teams in each list is 8 divided by 2 to the power of round number minus 1. 8 teams in round 1, 4 teams in round 2, 2 teams in round 3
    # num_teams = int(32/(2**(round-1)))
    score = 0   # score holder
    per_win = 10 * (2 ** (round - 1)) # Round 1: 10pts, Round 2: 20pts, Round3: 40pts, etc

    # Iterate through each team in the predicted winner list, and if that seed number is in actual winners, add to score
    for team in pred_winner:
        if team in actual_winner:
            score += per_win

    return score

def predict_winners(bracket, tourney_dict, model):
    """
    :param bracket: Bracket of teams to put through ML model
    :return: The winners predicted by the machine learning algorithm
    """
    # This will hold winners of each match for a particular region
    winners = []

    index = 0
    # go through bracket of teams
    while index < len(bracket):
        logging.debug("Index is {}".format(index))
        stats = get_matchup_data(tourney_dict[bracket[index]], tourney_dict[bracket[index + 1]], kp_df)

        stats = Variable(stats)
        winner = model(stats)

        # if winner is close to 1, team 2 is winner
        if winner > 0.5:
            winners.append(bracket[index + 1])

        # if winner is close to zero, team 1 is winner
        elif winner < 0.5:
            winners.append(bracket[index])

        else:
            print("NO CLEAR WINNER, CHECK MODEL HERE")
            print("Teams: {} vs {}".format(tourney_dict[bracket[index]],
                                           tourney_dict[bracket[index + 1]]))
            print("Predicted value is {}".format(winner))

        index += 2

    return winners

def create_64_bracket(regions):
    """

    :param regions: List of list, each list being 16 values
    :return:
    """

    bracket_64 = []
    # Cycle through each region list
    for index in range(4):
        region = regions[index]

        for seed in region:
            # Each seed is a function of its region, ie region 1 seed 1 is new_seed 1, region 3 seed 1 is new_seed 33
            # The actual function is (seed # + (16 * (region # - 1)). Index is already equal to region # - 1, since "regions" list starts at index 0
            # Add value to bracket_64
            bracket_64.append(seed + (16*index))

    return bracket_64

if __name__ == '__main__':

    logging.basicConfig(level = logging.DEBUG)
    year = 2018

    model = training.Lin_Relu(20)
    model = test_model.load_model(".\\Models\\500_epochs", model)
    # model = test_model.load_model(".\\Models\\5_epochs", model)

    kp_df = kenpom_df(".\\Training_Data\\KenPom_Complete.csv", year)

    # Can't simply build 64 team bracket, need four sections of 16 team brackets that play against each other
    region1 = generate_tournament(16)

    regions = [region1]*4

    bracket_64 = create_64_bracket(regions)

    # tourney_dict functions as match data, while maintaining easily accessible team information based on region and seed
    tourney_dict = match_teams_seeds(".\\Training_Data\\Match_Data.csv", year)

    total_score = 0

    for round in range(1,7):
        # Get list of predicted winners for each round
        pred_win = predict_winners(bracket_64, tourney_dict, model)
        logging.debug("Predicted winners: {}".format(pred_win))

        real_win = actual_bracket(year, round, tourney_dict)
        logging.debug("Real Bracket Winners are {}".format(real_win))

        if len(pred_win) == 1:
            logging.info("Predicted winning team is {}\nReal winning team is {}".
                         format(tourney_dict[pred_win[0]],tourney_dict[real_win[0]]))
        # Add this round's score to the total score
        rnd_score = score_per_round(round, pred_win, real_win)
        total_score += rnd_score
        logging.debug("Round {} score is {}".format(round, rnd_score))

        bracket_64 = pred_win

    logging.info("Final score is: {}".format(total_score))

