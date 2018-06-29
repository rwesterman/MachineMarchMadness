from torch.utils.data import DataLoader

from dataload import Match_Winners
from training import linrelu_train, linsig_train, binclass_train

import logging

def test_sig(filepath, num_epochs, model_choice, test_diff = False):
    dataset = Match_Winners(r"./Training_Data/Training_Set.csv", test_diff)
    num_inputs = dataset.x_data.shape[1]
    training_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers = 4)

    if model_choice == 1:
        linsig_train(training_loader, num_inputs, num_epochs)
    elif model_choice == 2:
        linrelu_train(training_loader, num_inputs, num_epochs)
    elif model_choice == 3:
        binclass_train(training_loader, num_inputs, num_epochs)

def epoch_input():
    print("Training model for March Madness.")
    print("There are currently 2 training models.\n1)Linear Sigmoid model\n2)ReLU model\n3)Binary Classifier")
    model_choice = int(input("Please enter 1, 2, or 3 for the model you'd like to use: "))
    if not (1 <= model_choice <= 3):
        input("Invalid entry. Please enter 1, 2, or 3 for the model you'd like to use: ")

    try:
        epochs = int(input("How many epochs should be run?: "))
    except (TypeError,ValueError) as e:
        print("Please enter a positive integer.")
        epochs = int(input("How many epochs should be run?: "))

    if epochs > 0:
        print("Running {} epochs".format(epochs))
        test_sig(r"./Training_Data/Training_Set.csv", epochs, model_choice)
    else:
        print("Defaulting to 100 epochs.")
        test_sig(r"./Training_Data/Training_Set.csv", 100, model_choice)

if __name__ == '__main__':
    # Setup output to log file and to stdout
    logging.basicConfig(filename = ".\\Models\\LossLog_with_elo.txt", level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Use this to ask user for number of epochs to run
    epoch_input()
