import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import training
from dataload import Match_Winners

def get_testdata():
    """

    :return: returns DataLoader object and number of inputs to the model
    """
    dataset = Match_Winners(r"./Training_Data/Training_Set.csv", test_diff=False)
    num_inputs = dataset.x_data.shape[1]
    training_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)


    return training_loader, num_inputs


def validate_sig(val_data, model):
    correct = 0
    total = 0

    for i, data in enumerate(val_data):
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)

        y_pred = model(inputs)
        # print("pred winner {} vs real winner {}".format(y_pred.data[0][0], labels))

        # Equation to check if the prediction is accurate
        # y - y_pred is correct if between -0.5 and 0.5 because
        # y = 1 - (0.5 or greater) < 0.5, y = 0 - (0.5 or less) > -0.5
        if -0.5 < (labels.data[0][0] - y_pred.data[0][0]) < 0.5:
            correct += 1

        total += 1

        print("Predicted winner is {:.1f}, actual winner is {}".format(y_pred.item(), labels.item()))

    accuracy = correct/total
    print("Percentage correct: {}".format(accuracy))
    return accuracy

def load_model(filepath, model):
    model.load_state_dict(torch.load(filepath))

    return model

def run_validation(model_path):

    training_loader, num_inputs = get_testdata()

    model = training.Lin_Sig(num_inputs)

    load_model(model_path, model)

    accuracy = validate_sig(training_loader, model)
