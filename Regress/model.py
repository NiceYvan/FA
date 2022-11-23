import torch
import torch.nn


class LogisticRegression(torch.nn.Module):
    def __init__(self, inputSize, numClasses, channels=3):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Sequential(
              torch.nn.Linear(inputSize*inputSize*channels, numClasses)
        )

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class Layer4NN(torch.nn.Module):
    def __init__(self, inputSize, numClasses, channels=3):
        super(Layer4NN, self).__init__()

        self.cnn_layer1 = torch.nn.Conv2d(
                            channels, 32, kernel_size=3, stride=1, padding=1)
        self.cnn_layer2 = torch.nn.Conv2d(
                            32, 32, kernel_size=3, stride=1, padding=1)

        self.fclayer1 = torch.nn.Linear(inputSize*inputSize*8, 128)
        self.fclayer2 = torch.nn.Linear(128, numClasses)

        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = torch.nn.ReLU()

        self.input = inputSize*inputSize

    def forward(self, x):
        x = self.relu(self.cnn_layer1(x))
        x = self.relu(self.cnn_layer2(x))
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = x.view(-1, self.input*8)
        x = self.relu(self.fclayer1(x))
        x = self.dropout2(x)
        x = self.fclayer2(x)
        return x
