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
        self.filepath = "..\\Training_Data\\Raw\\Elo_Data.csv"
        self.columns = ["Year", "Schl", "Opp", "Outcome"]

    def get_df(self):
        # Reduce columns of dataframe to team, opponent, and winner
        df = pd.read_csv(self.filepath,header = 0, usecols = self.columns)
        df = df[df["Year"] == self.year]
        return df


def write_elo_csv(ranking_list):
    with open("elo_rankings.csv", 'a') as f:
        for year, team, rank in ranking_list:
            f.write("{},{},{}\n".format(year, team, rank))



if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)

    year = 2018
    data_elo = DataElo(year)
    df = data_elo.get_df()

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

    for player in game.players:
        logging.debug("{} has played {} games".format(player.name, player.games_played))

    write_elo_csv(team_list)





