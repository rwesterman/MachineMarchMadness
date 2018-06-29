import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import logging
import time

from dataload import Match_Winners

# Taken from example: https://gist.github.com/santi-pdp/d0e9002afe74db04aa5bbff6d076e8fe
class Binary_Classifier(nn.Module):

    def __init__(self, num_inputs):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        a1 = self.fc1(x)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

class Lin_Sig(nn.Module):

    def __init__(self, num_inputs):
        super().__init__()

        # Making three-deep neural network with 12 inputs
        # 12 inputs in x_data. Arbitrary output 8
        self.lin1 = nn.Linear(num_inputs, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 60)
        self.lin4 = nn.Linear(60, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x.view()
        out1 = self.sigmoid(self.lin1(x))
        out2 = self.sigmoid(self.lin2(out1))
        out3 = self.sigmoid(self.lin3(out2))
        y_pred = self.sigmoid(self.lin4(out3))

        return y_pred


class Lin_Relu(nn.Module):

    def __init__(self, num_inputs):
        super().__init__()

        # Making three-deep neural network with 12 inputs
        # 12 inputs in x_data. Arbitrary output 8
        self.lin1 = nn.Linear(num_inputs, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 60)
        self.lin4 = nn.Linear(60, 100)
        self.lin5 = nn.Linear(100, 80)
        self.lin6 = nn.Linear(80, 60)
        self.lin7 = nn.Linear(60, 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, x):

        out1 = self.relu(self.lin1(x))
        out2 = self.relu(self.lin2(out1))
        out3 = self.relu(self.lin3(out2))
        out4 = self.relu(self.lin4(out3))
        out5 = self.relu(self.lin5(out4))
        out6 = self.relu(self.lin6(out5))
        y_pred = self.sigmoid(self.lin7(out6))

        return y_pred

def binclass_train(training_loader, num_inputs, num_epochs):
    model = Binary_Classifier(num_inputs)

    # set up Cuda device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Testing different Loss functions
    criterion = nn.BCELoss(size_average= True)
    # criterion = nn.MSELoss(size_average=True)
    # criterion = nn.CrossEntropyLoss(size_average=True)
    # criterion = nn.KLDivLoss(size_average=True)

    # Testing different optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    # optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    start_time = time.time()

    for epoch in range(num_epochs):
        for i, data in enumerate(training_loader):
            # get the inputs and results
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)


            y_pred = model(inputs)

            # Compute and print loss
            loss = criterion(y_pred, labels)

            # Print loss for every 10th batch
            if i == 0:
                current_time = time.time()
                elapsed_time = current_time - start_time
                epochs_remaining = num_epochs - epoch
                time_per_epoch = elapsed_time/epoch
                logging.info("{} seconds elapsed\t\testimated time remaining is {}".format(elapsed_time, time_per_epoch * epochs_remaining))
                logging.info("Epoch: {}, batch #: {}, loss: {:.5f}".format(epoch, i, loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    save_path = ".\\Models\\{}_epochs_binary_classifier".format(num_epochs)
    torch.save(model.state_dict(), save_path)
    print("Saving model to {}".format(save_path))

def linrelu_train(training_loader, num_inputs, num_epochs):
    model = Lin_Relu(num_inputs)

    # Testing different Loss functions
    criterion = nn.MSELoss(size_average=True)
    # criterion = nn.CrossEntropyLoss(size_average=True)
    # criterion = nn.KLDivLoss(size_average=True)

    # Testing different optimizers
    optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.01)
    # optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for i, data in enumerate(training_loader):
            # get the inputs and results
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            y_pred = model(inputs)

            # Compute and print loss
            loss = criterion(y_pred, labels)

            # Print loss for every 10th batch
            if i % 8 == 0 and epoch % 2 == 0:
                logging.info("Epoch: {}, batch #: {}, loss: {:.5f}".format(epoch, i, loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # dataset = Match_Winners(r"./Training_Data/Training_Set.csv", test_diff = False)
    # num_inputs = dataset.x_data.shape[1]
    # training_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 2)
    # accuracy = validate_sig(training_loader, model)
    # with open("Accuracy_results.txt", 'a') as f:
    #     f.write("{} inputs, {} Epochs, {} Accuracy\n".format(num_inputs, num_epochs, accuracy))


    save_path = ".\\Models\\{}_epochs_relu".format(num_epochs)
    torch.save(model.state_dict(), save_path)
    print("Saving model to {}".format(save_path))


def linsig_train(training_loader, num_inputs, num_epochs):
    model = Lin_Relu(num_inputs)

    # Testing different Loss functions
    # criterion = nn.BCELoss(size_average= True)
    criterion = nn.MSELoss(size_average=True)
    # criterion = nn.CrossEntropyLoss(size_average=True)
    # criterion = nn.KLDivLoss(size_average=True)

    # Testing different optimizers
    optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.01)
    # optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for i, data in enumerate(training_loader):
            # get the inputs and results
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            y_pred = model(inputs)

            # Compute and print loss
            loss = criterion(y_pred, labels)

            # Print loss for every 10th batch
            if i % 8 == 0 and epoch % 2 == 0:
                logging.info("Epoch: {}, batch #: {}, loss: {:.5f}".format(epoch, i, loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    save_path = ".\\Models\\{}_epochs_linsig".format(num_epochs)
    torch.save(model.state_dict(), save_path)
    print("Saving model to {}".format(save_path))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # We do have a cuda device. Now to set up network to run with CUDA