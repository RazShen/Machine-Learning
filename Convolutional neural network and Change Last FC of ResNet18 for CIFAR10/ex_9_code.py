import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 40
LEARN_RATE = 0.065
EPOCHS = 10


class ModelBuilder(object):

    def __init__(self):

        tran = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

        """
        For cifar10 model
        """
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

        """
        For resnet
        """
        # train_dataset = datasets.CIFAR10(root='./data', train=True, transform=tran, download=True)
        # test_dataset = datasets.CIFAR10(root='./data', train=False, transform=tran)
        # Define the indices
        indices = list(range(len(train_dataset)))  # start with all the indices in training set
        split = int(len(train_dataset) * 0.2)  # define the split size

        # Random, non-contiguous split
        validation_idx = np.random.choice(indices, size=split, replace=False)
        train_idx = list(set(indices) - set(validation_idx))

        # define our samplers -- we use a SubsetRandomSampler because it will return
        # a random subset of the split defined by the given indices without replacement
        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)

        # define loaders
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        self.validation_loader = DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        # initialize model
        """
        Model 1 initialization
        """
        self.model = BasicModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARN_RATE)

        """
        Resnet model initialzation
        """

        # self.model = torchvision.models.resnet18(pretrained=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 10)
        # # initialize optimizer for the model
        # self.optimizer = optim.SGD(self.model.fc.parameters(), lr=LEARN_RATE)

        self.criterion = nn.CrossEntropyLoss()

        # initialize the dictionaries for the plot
        self.validation_print_dict = {}
        self.train_print_dict = {}
        # self.test_file = pickle.load(open('test.pickle', 'rb'))

    def train_validate_test(self):
        """
        This method trains the model and validate it, and then test the model & print the results
        into a graph.
        :return:
        """
        for epoch in range(1, EPOCHS + 1):
            self.train(epoch)
            self.validation(epoch)
        self.test()
        self.print_results()

    def print_results(self):
        """
        This method draws the graph by using the validation and train epoch-loss dictionaries
        :return:
        """
        norm_line, = plt.plot(self.validation_print_dict.keys(), self.validation_print_dict.values(), "red",
                              label='Validation loss')
        trained_line, = plt.plot(self.train_print_dict.keys(), self.train_print_dict.values(), "black",
                                 label='Train loss')
        plt.legend(handler_map={norm_line: HandlerLine2D()})
        plt.show()

    def validation(self, epoch_num):
        """
        This method goes through the data, get the model output and validate it by using negative log likelihood loss
        and softmax and also updates the dictionary and print the results.
        :param epoch_num: number of epoch we're in
        :return:
        """
        self.model.eval()
        validation_loss = 0
        correct = 0
        for data, target in self.validation_loader:
            output = self.model(data)
            validation_loss += self.criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        validation_loss /= len(self.validation_loader)
        print('\n Validation epoch number :{} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_num, validation_loss, correct, len(self.validation_loader),
            100. * correct / len(self.validation_loader)))
        self.validation_print_dict[epoch_num] = validation_loss

    def test(self):
        """
        This method goes through the data, get the model output and validate it by using negative log likelihood loss
        and softmax and also print the results for the test and write the predictions to the test.pred file.
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        y_pred_for_mat = []
        y_tag_for_mat = []
        prediction_file = open("test.pred", 'w')
        for data, target in self.test_loader:
            output = self.model(data)
            y_tag_for_mat.append(target.item())
            test_loss += self.criterion(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            y_pred_for_mat.append(pred.item())
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            prediction_file.write(str(pred.item()) + "\n")
        conf_mat = confusion_matrix(y_tag_for_mat, y_pred_for_mat)
        print(conf_mat)
        prediction_file.close()
        test_loss /= len(self.test_loader)
        print('\n Test Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader), 100. * correct / len(self.test_loader)))

    def train(self, epoch):
        """
        This method goes through the data, get the model output and validate it by using negative log likelihood loss
        and softmax and also train the model for every batch of examples we have looped through using our optimizer.
        :param epoch: number of epoch we're in
        :return:
        """
        self.model.train()
        correct = 0
        train_loss = 0
        for batchidx, (data, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            # negative log likelihood loss
            loss = self.criterion(output, labels)
            # calculate gradients
            train_loss += loss
            loss.backward()
            # update parameters
            self.optimizer.step()
        train_loss /= len(self.train_loader)
        print('\n Train epoch number: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss, correct, len(self.train_loader) * BATCH_SIZE,
                                        (100. * correct) / (len(self.train_loader) * BATCH_SIZE)))
        self.train_print_dict[epoch] = train_loss


class BasicModel(nn.Module):
    """
    Convolution layer with batch normalization layers, max pool, 3 fully connected layers and 2 convolution layers
    """

    def __init__(self):
        """
        Neural network inherits from nn.Module that has 2 hidden layers, W1,b1,W2,b2,W3,b3. and 2 batch normalization
        parameters.
        :param image_size: size of image
        """
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 for input (rgb) 6 for number of filters, 5 for size of filter.
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.batch = nn.BatchNorm2d(6)
        self.batch1 = nn.BatchNorm1d(120)
        self.batch2 = nn.BatchNorm1d(84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.batch1(self.fc1(x)))
        x = F.relu(self.batch2(self.fc2(x)))
        x = self.fc3(x)
        return x


def main():
    """
    Initializes model builder, train, validate and then test the model.
    :return:
    """
    my_obj = ModelBuilder()
    my_obj.train_validate_test()


if __name__ == "__main__":
    main()
