from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from dataload import Match_Winners
from training import linrelu_train, linsig_train, binclass_train
from test_model import validate_sig

import logging

def plot_loss(x_vals, y_vals):

    # plt.scatter(x_vals, y_vals)
    plt.plot(x_vals, y_vals)
    plt.show()

def test_sig(filepath, num_epochs, model_choice, test_diff = False):
    # Testing now with normalized data
    dataset = Match_Winners(r"./Training_Data/Training_Set_Normalized.csv", test_diff)
    num_inputs = dataset.x_data.shape[1]
    training_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers = 4)

    if model_choice == 1:
        loss_list = linsig_train(training_loader, num_inputs, num_epochs)
    elif model_choice == 2:
        loss_list = linrelu_train(training_loader, num_inputs, num_epochs)
    elif model_choice == 3:
        loss_list = binclass_train(training_loader, num_inputs, num_epochs)

    x_vals = [x for x in range(num_epochs)]
    plot_loss(x_vals, loss_list)


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
    # epoch_input()

    test_sig(r"./Training_Data/Training_Set.csv", 50, 2)


