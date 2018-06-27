import pandas as pd

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
        pass

