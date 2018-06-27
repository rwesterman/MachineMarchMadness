import pandas as pd
from random import randint
import logging

class Data_Frames():
    def __init__(self, kp_path = "..\\Training_Data\\KenPom_Complete.csv", bpi_path = "..\\Training_Data\\ESPN_BPI.csv",
                 elo_path = "..\\Training_Data\\elo_rankings.csv", match_path = "..\\Training_Data\\Match_Data.csv"):
        self.kp_path = kp_path
        self.bpi_path = bpi_path
        self.elo_path = elo_path
        self.match_path = match_path

    def get_kp_df(self):
        """Returns full kp_df from the given filepath"""
        return pd.read_csv(self.kp_path)

    def get_kp_df_year(self, year):
        """Returns kp_df limited to a specific year"""
        kp_df = self.get_kp_df()
        return kp_df[kp_df["Year"] == year]

    def get_bpi_df(self):
        """Returns full bpi_df from given filepath"""
        return pd.read_csv(self.bpi_path, encoding= "latin-1")

    def get_bpi_df_year(self, year):
        bpi_df = self.get_bpi_df()
        return bpi_df[bpi_df["Year"] == year]

    def get_elo_df(self):
        """Returns full kp_df from the given filepath"""
        return pd.read_csv(self.elo_path)

    def get_elo_df_year(self, year):
        """Returns elo_df limited to a specific year"""
        elo_df = self.get_elo_df()
        return elo_df[elo_df["Year"] == year]

    def get_match_df(self):
        """Returns match_df of all tournament match data"""
        return pd.read_csv(self.match_path)

    def get_match_df_year(self, year):
        """Returns match_df of tournament matches for a single year"""
        match_df = self.get_match_df()
        return match_df[match_df["Year"] == year]

    def combine_dfs(self, match_df, kp_df, bpi_df, elo_df):
        """Combines all existing dataframes into single matchup dataframe"""
        # Todo: Figure out how to merge all dataframes along the "Team" column, then match them up with match_df, and output to file?
        merge_cols = ["Year", "Team"]

        # Drop columns from bpi_df that aren't needed in merge
        bpi_df = bpi_df.drop(["W-L", "Rk", "Conf"], axis=1)

        # Start by merging bpi_df with kp_df
        comb_df = kp_df.merge(bpi_df, on = merge_cols)

        comb_df = comb_df.merge(elo_df, on = merge_cols)

        # Don't need to write this to csv yet
        # comb_df.to_csv("DELETE.csv", ',', mode = 'w', index = False)

        return comb_df


    def associate_data(self, match_data, comb_df, kenpom_df, bpi_df):
        # Todo: Get Combination dataframe from self.combine_dfs above, and pass it to this function instead of kenpom_df and bpi_df
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

        logging.debug(f"Team1 is {score_dict['Team1']}, Team2 is {match_data['Team2']}")

        # Reduce all dataframes to two teams from given year
        kp_df = kenpom_df.loc[kenpom_df["Year"] == score_dict["Year"]]
        bpi_df = bpi_df[bpi_df["Year"] == score_dict["Year"]]

        # Drop conflicting columns from bpi dataframe
        bpi_df = bpi_df.drop(["W-L", "Rk", "Conf"], axis=1)

        # Todo: Merge dataframes across all years and teams, instead of just two teams in one year
        # Get the kenpom data for each of the teams in the given year (teams is two row dataframe)
        teams = kp_df[(kp_df["Team"] == match_data["Team"]) | (kp_df["Team"] == match_data["Team.1"])]
        teams2 = bpi_df[(bpi_df["Team"] == match_data["Team"]) | (bpi_df["Team"] == match_data["Team.1"])]

        # Want to randomize the team so the efficiency isn't always higher for team 1
        teamnum = randint(1, 2)

        # Remember to use df.merge instead of join if both dataframes have same column name
        comb_df = teams.merge(teams2, on="Team", how="outer")

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
            final_data.update(self.organize_data(data_dict, teamnum))

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

    def organize_data(self, data_dict, teamnum):
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

        return output_dict


if __name__ == '__main__':
    datadog = Data_Frames()
    kp_df = datadog.get_kp_df()
    bpi_df = datadog.get_bpi_df()
    elo_df = datadog.get_elo_df()
    match_df = datadog.get_match_df()

    datadog.combine_dfs(match_df, kp_df, bpi_df, elo_df)