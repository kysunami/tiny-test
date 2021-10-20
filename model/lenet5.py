import torch.nn as nn
import numpy as np


class LeNet5(nn.Module):

    def __init__(
        self,
        channels=3,
        class_count=10,
        act='relu'
    ):

        super(LeNet5, self).__init__()

        conv_layers = []
        fc_layers = []

        # Conv1 -> out_ch = 6, filter = 5x5, stride = 1
        conv_layers.append(nn.Conv2d(channels, 6, 5))

        # activation
        conv_layers.append(self.get_activation(act))

        # Pool2 -> filter = 2x2, stride = 2
        conv_layers.append(nn.MaxPool2d(2, 2))

        # Conv3 -> out_ch = 16, filter = 5x5, stride = 1
        conv_layers.append(nn.Conv2d(6, 16, 5))

        # activation
        conv_layers.append(self.get_activation(act))

        # Pool4 -> filter = 2x2, stride = 2
        conv_layers.append(nn.MaxPool2d(2, 2))

        # Conv5 -> out_ch = 120, filter = 5x5
        conv_layers.append(nn.Conv2d(16, 120, 5))

        # activation
        conv_layers.append(self.get_activation(act))

        # FC6 -> in_ch = 120, out_ch = 84
        fc_layers.append(nn.Linear(120, 84))

        # activation
        fc_layers.append(self.get_activation(act))

        # FC7 -> in_ch = 84, out_ch = class_count
        fc_layers.append(nn.Linear(84, class_count))

        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*fc_layers)

    def get_activation(self, act='relu'):

        activation = nn.ReLU(inplace=True)
        if act == 'sigmoid':
            activation = nn.Sigmoid()
        elif act == 'tanh':
            activation = nn.Tanh()

        return activation

    def forward(self, x):
        y = self.conv(x)
        y = y.view(-1, 120)
        return self.fc(y)





def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}

