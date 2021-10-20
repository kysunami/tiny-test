import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#from ib_layers import *

class LeNet5m(nn.Module):

    def __init__(self,kml=900,masking=False):
        super(LeNet5m, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)#1, 20, 5, 1)
        self.maxp=nn.MaxPool2d(2,2) #max_pool2d(2,2)
        #self.ib1= InformationBottleneck(20, kl_mult= kml,masking=masking )
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.maxp2=nn.MaxPool2d(2,2)
        #self.ib2= InformationBottleneck(50, kl_mult= kml*2,masking=masking )
        self.fc1 = nn.Linear(5*5*50, 500)#4*4*50, 500)
        #self.ib1f= InformationBottleneck(500, kl_mult= kml*10,masking=masking )
        self.fc2 = nn.Linear(500, 200)

    def get_activation(self, act='relu'):

        activation = nn.ReLU(inplace=True)
        if act == 'sigmoid':
            activation = nn.Sigmoid()
        elif act == 'tanh':
            activation = nn.Tanh()

        return activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("\n", x.size()[3])
        x = F.max_pool2d(x, 2, 2)
        #x=self.ib1(x)
        #print("\n", x.size()[3])
        x = F.relu(self.conv2(x))

        #self.g1= x.size()[3]
        x = F.max_pool2d(x, 2, 2)
        #x=self.ib2(x)
        x = x.view(-1, 5*5*50)#4*4*50)
        #print("\n", x.size()[3])
        x = F.relu(self.fc1(x))
        #x=self.ib1f(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)





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

