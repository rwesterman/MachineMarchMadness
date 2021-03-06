from setup.elo import Implementation
import pandas as pd
from pprint import pprint
import csv
import logging

# Implementation is a class to track Elo data of teams

# Todo: Create csv or excel file for all matches from each season 2002-2018

# Create dataframe of every match from one year
class DataElo():
    def __init__(self, year):
        self.year = year
        self.filepath = "..\\Training_Data\\Raw\\Elo_Data_Winners.csv"
        self.columns = ["Year", "Schl", "Opp", "Outcome"]

    def get_df(self):
        # Reduce columns of dataframe to team, opponent, and winner
        df = pd.read_csv(self.filepath,header = 0, usecols = self.columns)
        df = df[df["Year"] == self.year]
        return df


def write_elo_csv(ranking_list):
    with open("..\\Training_Data\elo_rankings.csv", 'a') as f:
        for year, team, rank in ranking_list:
            f.write("{},{},{}\n".format(year, team, rank))


def determine_elo_ranks(year):
    data_elo = DataElo(year)
    df = data_elo.get_df()

    # Need to make a new Implementation object for each year
    game = Implementation()

    allPlayers = set()

    # itertuples is extremely fast way to iterate through rows of dataframe
    for year, schl, opp, outcome in df.itertuples(index = False):

        # add both teams to allPlayers set
        allPlayers.add(schl)
        allPlayers.add(opp)

        # Check if schl already exists in game object, if not, game.getPlayer(schl) will return None
        if not game.contains(schl):
            # add schl to game object if not already present
            game.addPlayer(schl)

        # Check if opp already exists in game object, if not, game.getPlayer(opp) will return None
        if not game.contains(opp):
            # add opp to game object if not already present
            game.addPlayer(opp)

        # Record the match with team1, team2, and winner = "W" (for team1 winning) or "L" (for team2 winning)
        # This will automatically adjust the Elo rankings for both teams
        game.recordMatch(schl, opp, outcome)

    # After all matches recorded, can use .getRankingsList() to see final rankings, and then output it to a csv file

    team_list = []
    for team in allPlayers:
        # print(team, game.getPlayerRating(team))
        team_rank = (year, team, game.getPlayerRating(team))
        team_list.append(team_rank)

    # for player in game.players:
        # game.log("info", "{} has played {} games".format(player.name, player.games_played))

    write_elo_csv(team_list)


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)

    # iterate through years 2003-2018
    for year in range(2003, 2019):
        determine_elo_ranks(year)






