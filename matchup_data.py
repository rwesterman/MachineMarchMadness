import pandas as pd


def get_match_data():
    # Get tournament results from data.world website.
    # https://data.world/michaelaroy/ncaa-tournament-results
    # df = pd.read_csv('https://query.data.world/s/xu365o2nhb4vvbqtm2moubtt63onr7', index_col = "Year")
    df = pd.read_csv('Training_Data\\Match_Data.csv', index_col = "Year")
    return df

def get_year_data(year, df):
    """
    Returns limited dataframe of just one year's matches
    :param year: int for year value
    :param df: pandas dataframe of full match data
    :return: abridged dataframe of one year's matches
    """
    year_data = df.loc[year]
    return year_data

def associate_data(match_data, kenpom_df, samples):
    """

    :param match_data: one-row dataframe of match results
    :param kenpom_df: full kenpom dataframe for given year
    :return: ??? Dict of team 1 data, team 2 data, scores, winner?
    """
    # Todo: tie together data on teams from kenpom rankings and match results
    # Access row x with df.iloc[x], then column name to get value
    # Since "Year" is index, need to use workaround shown below with df.index.tolist()[x]
    match_list = []
    for x in range(samples):
        # print(x)
        match_dict = {"Team1": match_data.iloc[x]["Team"],
                      "Score1": match_data.iloc[x]["Score"],
                      "Team2": match_data.iloc[x]["Team.1"],
                      "Score2": match_data.iloc[x]["Score.1"],
                      "Year": match_data.index.tolist()[x],
                      "Seed1": match_data.iloc[x]["Seed"],
                      "Seed2": match_data.iloc[x]["Seed.1"]}
        match_list.append(match_dict)

    for x in range(samples):
        print(f"Team1 is {match_list[x]['Team1']}, Team2 is {match_list[x]['Team2']}")
        # Reduce kenpom df to only two teams from given year
        # print(kenpom_df.loc[match_dict["Year"]])
        kp_df = kenpom_df.loc[match_list[x]["Year"]]
        print(kp_df[(kp_df["Team"] == match_list[x]["Team1"]) | (kp_df["Team"] == match_list[x]["Team2"])])


def import_kenpom_df():
    kp_comp = pd.read_csv("Training_Data\\KenPom_Complete.csv", index_col = "Year")
    return kp_comp

if __name__ == '__main__':
    # TODO: Standardize team names across data. Ex. NC State vs North Carolina State, Pennsylvania vs Penn
    # In general, follow KenPom methodology or do this programatically so it can be entered for new 2019 data
    # Ole Miss vs Mississippi
    samples = 10
    df = get_match_data()
    data_2002 = get_year_data(2002, df)
    kp_comp = import_kenpom_df()
    datum = data_2002.sample(samples)
    associate_data(datum, kp_comp, samples)
