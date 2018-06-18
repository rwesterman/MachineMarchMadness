import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataload import Match_Winners


class Lin_Sig(torch.nn.Module):

    def __init__(self, num_inputs):
        super().__init__()

        # Making three-deep neural network with 12 inputs
        # 12 inputs in x_data. Arbitrary output 8
        self.lin1 = torch.nn.Linear(num_inputs, 200)
        self.lin2 = torch.nn.Linear(200, 100)
        self.lin3 = torch.nn.Linear(100, 60)
        self.lin4 = torch.nn.Linear(60, 100)
        self.lin5 = torch.nn.Linear(100, 80)
        self.lin6 = torch.nn.Linear(80, 60)
        self.lin7 = torch.nn.Linear(60, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        out1 = self.relu(self.lin1(x))
        out2 = self.relu(self.lin2(out1))
        out3 = self.relu(self.lin3(out2))
        out4 = self.relu(self.lin4(out3))
        out5 = self.relu(self.lin5(out4))
        out6 = self.relu(self.lin6(out5))
        y_pred = self.sigmoid(self.lin7(out6))

        return y_pred


def linsig_train(training_loader, num_inputs, num_epochs):
    model = Lin_Sig(num_inputs)

    # Testing different Loss functions
    criterion = torch.nn.MSELoss(size_average=True)
    # criterion = torch.nn.CrossEntropyLoss(size_average=True)
    # criterion = torch.nn.KLDivLoss(size_average=True)

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
            # if i % 10 == 0 and epoch % 5 == 0:
            print("Epoch: {}, batch #: {}, loss: {:.5f}".format(epoch, i, loss.item()))

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


    save_path = ".\\Models\\{}_epochs".format(num_epochs)
    torch.save(model.state_dict(), save_path)
    print("Saving model to {}".format(save_path))