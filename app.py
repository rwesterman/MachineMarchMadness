from torch.utils.data import DataLoader

from dataload import Match_Winners
from training import linsig_train

import test_model
import logging

def test_sig(filepath, num_epochs, test_diff = False):
    dataset = Match_Winners(r"./Training_Data/Training_Set.csv", test_diff)
    num_inputs = dataset.x_data.shape[1]
    training_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers = 4)
    linsig_train(training_loader, num_inputs, num_epochs)

def epoch_input():
    try:
        epochs = int(input("How many epochs should be run?: "))
    except (TypeError,ValueError) as e:
        print("Please enter a positive integer.")
        epochs = int(input("How many epochs should be run?: "))

    if epochs > 0:
        print("Running {} epochs".format(epochs))
        test_sig(r"./Training_Data/Training_Set.csv", epochs)
    else:
        print("Defaulting to 100 epochs.")
        test_sig(r"./Training_Data/Training_Set.csv", 100)

if __name__ == '__main__':
    # Setup output to log file and to stdout
    logging.basicConfig(filename = ".\\Models\\LossLog.txt", level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Load in the match data
    # Setting test_diff to true means we're only looking at one input, "Eff_Diff"
    # test_sig(r"./Training_Data/Training_Set.csv", 5)
    # test_sig(r"./Training_Data/Training_Set.csv", 100)
    # test_sig(r"./Training_Data/Training_Set.csv", 2000)

    # Use this to ask user for number of epochs to run
    epoch_input()

    # test_model.run_validation(".\\Models\\5000_epochs")

