import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class Match_Scores(Dataset):
    def __init__(self, filepath, test_diff = False):
        """
        Loads dataset with Score_Diff column as the output value
       :param filepath: Path to data file to load
       :param test_diff: Boolean. False if inputs are two columns (Eff1, Eff2), or True if one column (Eff Diff)
       """
        with open(filepath, 'r') as f:
            self.df = pd.read_csv(f, delimiter = ',')

        if test_diff:
            # x_data is all rows after row 1, only eff_diff column
            self.x_np = np.asarray(self.df.iloc[1:, [-3]], dtype=np.float32)
        else:
            # x_data is all rows after row 1, columns 2 through 14 non-inclusive
            self.x_np = np.asarray(self.df.iloc[1:, 2:14],dtype = np.float32)
        # y_data is all rows after row 1, only the second to last column (Score_Diff)
        self.y_np = np.asarray(self.df.iloc[1:,[-2]], dtype = np.float32)

        # xy_data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # set length attribute to get data. Subtract one to account for header row
        self.len = self.df.shape[0] - 1

        self.x_data = torch.from_numpy(self.x_np)
        self.y_data = torch.from_numpy(self.y_np)
        # x_data is all rows after row 1, columns 2 through 13 inclusive
        # self.x_data = torch.from_numpy(xy_data[1:,2:13])
        # y_data is all rows after row 1, only the last column (winner)
        # self.y_data = torch.from_numpy(xy_data[1:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Match_Winners(Dataset):
    def __init__(self, filepath, test_diff = False):
        """
        Loads dataset with Winner column as the output value
        :param filepath:
        :param test_diff: Boolean. False if inputs are two columns (Eff1, Eff2), or True if one column (Eff Diff)
        """
        with open(filepath, 'r') as f:
            self.df = pd.read_csv(f, delimiter = ',')

        if test_diff:
            # x_data is all rows after row 1, only eff_diff column
            self.x_np = np.asarray(self.df.iloc[1:, [-3]],dtype = np.float32)
        else:
            # x_data is all rows after row 1, columns 2 through 4 non-inclusive
            self.x_np = np.asarray(self.df.iloc[1:, 2:14], dtype=np.float32)
        # y_data is all rows after row 1, only the second to last column (Winner)
        self.y_np = np.asarray(self.df.iloc[1:,[-1]], dtype = np.float32)

        # xy_data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        # set length attribute to get data. Subtract one to account for header row
        self.len = self.df.shape[0] - 1

        self.x_data = torch.from_numpy(self.x_np)
        self.y_data = torch.from_numpy(self.y_np)
        # x_data is all rows after row 1, columns 2 through 13 inclusive
        # self.x_data = torch.from_numpy(xy_data[1:,2:13])
        # y_data is all rows after row 1, only the last column (winner)
        # self.y_data = torch.from_numpy(xy_data[1:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len