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
    :return: Dict of dicts. Top level dict keys are region number, and the values are
    keys for seeds with the team's name associated
    """
    df = pd.read_csv(match_path, index_col="Year")
    df = df.loc[year]

    # All tournament teams are listed in Round 1
    df = df[df["Round"] == 1]

    tourney_dict = defaultdict(str)

    # Iterate through every row of the Match_Data.csv dataframe
    for row in df.itertuples():
        seed_dict = {}
        # row indexing - 1: Round, 2: Region Number, 3: Region Name, 4: Seed, 5: Score,
        # 6: Team, 7: Team.1, 8: Score.1, 9: Seed.1

        # Add team1's seed as key, team name as value
        seed_dict[row[4]] = row[6]
        # Add team2's seed as key, team name as value
        seed_dict[row[9]] = row[7]

        # Addresses KeyError by providing default value if the key doesn't exist already
        tourney_dict[row[2]] = tourney_dict.get(row[2], seed_dict)
        tourney_dict[row[2]].update(seed_dict)

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
        # Only excecuted on first pass??
        else:
            new_matches += [[team_or_match, num_teams + 1 - team_or_match]]
    return new_matches

def flatten_list( matches ):
    """Takes a list of lists, log2(num_teams)+1 lists deep, then flattens them recursively to single level"""
    teamlist = []
    for team_or_match in matches:
        if type(team_or_match) == type([]):
            teamlist += flatten_list( team_or_match )
        else:
            teamlist += [team_or_match]
    return teamlist

def generate_tournament(num_teams):
    """Generates a tournament?"""

    # The number of rounds can be calculated as log base 2 of number of teams
    num_rounds = math.log(num_teams, 2)

    # Catch exceptions if log2(num_teams) is not an integer
    if num_rounds != math.trunc( num_rounds ):
        raise ValueError( "Number of teams must be a power of 2" )

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

def actual_bracket(year, round):
    """

    :param year: year of tournament (int)
    :param round: round of tournament (int)
    :return: winners for given year and round
    """
    # Currently only have 2018 winner results
    # Dictionary of dictionaries, ordered by round number, then by region,
    # eg. year_2018[1][1] is winners from region 1 in round 1 of year 2018
    year_2018 ={1:
                    {1: [16, 9, 13, 5, 2, 7, 3, 11], 2: [1, 9, 4, 5, 2, 7, 3, 6],
                    3:[1, 9, 13, 5, 2, 10, 3, 6], 4:[1, 8, 4, 5, 2, 7, 3, 11]},
                2:
                    {1: [9, 5, 7, 11], 2: [9, 4, 7, 3], 3: [1, 5, 2, 3], 4: [1, 5, 2, 11]},
                3:
                    {1: [9, 11], 2: [9, 3], 3: [1, 3], 4: [1, 2]},
                4:
                    {1: [11], 2: [3], 3:[1], 4:[1]}}

    choices = {2018: year_2018}
    if not year in choices:
        raise ValueError("No data for given year")
    else:
        return choices[year][round]

def score_per_round(round, pred_winner, actual_winner):
    # number of teams in each list is 8 divided by 2 to the power of round number minus 1. 8 teams in round 1, 4 teams in round 2, 2 teams in round 3
    num_teams = int(8/(2**(round-1)))
    score = 0   # score holder
    per_win = 10 * (2 ** (round - 1)) # Round 1: 10pts, Round 2: 20pts, Round3: 40pts, etc
    for region in range(1,5):

        # walk through list of each region and tally the score
        for index in range(num_teams):
            # If the predicted winner is correct, add to the score total
            if pred_winner[region][index] == actual_winner[region][index]:
                score += per_win

    return score

def predict_winners(regions, tourney_dict):
    """
    :param regions: list of lists, where each element of interior list is seeds still in competition, and the list itself represents a region
    :return: The winners predicted by the machine learning algorithm
    """

    model = training.Lin_Sig(12)
    model = test_model.load_model(".\\Models\\5000_epochs", model)

    # This will hold winners of each match for a particular region
    winners = {1: [], 2: [], 3: [], 4: []}  # Dict of lists for each region's winner

    index = 0
    while index < len(regions[0]):

        # iterate through the four regions.
        # reg_num + 1 gives value of region, used to access tourney_dict for that region
        for reg_num, region in enumerate(regions):

            # AS LONG AS I PASS get_matchup_data the two team names, the seeds should be unimportant
            # return stats for matchup against two teams (index team and index+1
            stats = get_matchup_data(tourney_dict[reg_num + 1][region[index]],
                                     tourney_dict[reg_num + 1][region[index + 1]], kp_df)
            stats = Variable(stats)
            winner = model(stats)

            # if winner is close to 1, team 2 is winner
            if winner > 0.6:
                winners[reg_num + 1].append(region[index + 1])

            # if winner is close to zero, team 1 is winner
            elif winner < 0.4:
                winners[reg_num + 1].append(region[index])

            else:
                print("NO CLEAR WINNER, CHECK MODEL HERE")
                print("Teams: {} vs {}".format(tourney_dict[reg_num + 1][region[index]],
                                               tourney_dict[reg_num + 1][region[index + 1]]))
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

    kp_df = kenpom_df(".\\Training_Data\\KenPom_Complete.csv", year)

    # Divide into four 16-team regions
    region1 = generate_tournament(16)

    regions = [region1]*4

    bracket_64 = create_64_bracket(regions)

    # tourney_dict functions as match data, while maintaining easily accessible team information based on region and seed
    tourney_dict = match_teams_seeds(".\\Training_Data\\Match_Data.csv", year)

    total_score = 0
    for round in range(1,5):
        # Get dictionary of predicted winners for the round
        pred_win = predict_winners(regions, tourney_dict)
        print("Predicted winners: {}".format(pred_win))

        real_win = actual_bracket(year, round)
        print("Real winners are: {}".format(real_win))
        # Add this round's score to the total score
        rnd_score = score_per_round(round, pred_win, real_win)
        total_score += rnd_score
        print("Round {} score is {}".format(round, rnd_score))

        for index in range(4):
            regions[index] = pred_win[index+1]

    # Todo: Add functionality for final four and championship rounds where teams play against other regions

    print("Final score is: {}".format(total_score))


# TODO: Attempt to rewrite much of this as a 64 team bracket so there are no issues with the final
# Generate the 64 teams from the 4 16 team lists



