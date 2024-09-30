import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self, n_classes = 10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, n_classes)

    def forward(self, x, return_features = False):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        f = self.fc2(x)
        x = F.relu(f)
        x = self.fc3(x)
        if return_features:
            return x, f
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

class LeNetDynamic(nn.Module):

    def __init__(self, n_classes = 10):
        super(LeNetDynamic, self).__init__()
        self.filter1 = 6
        self.filter2 = 16
        self.neurons1 = 120
        self.neurons2 = 84
        self.neurons3 = n_classes
        self.conv1 = nn.Conv2d(1, self.filter1, 5, padding=2)
        self.conv2 = nn.Conv2d(self.filter1, self.filter2, 5)
        self.fc1   = nn.Linear(self.filter2*5*5, self.neurons1)
        self.fc2   = nn.Linear(self.neurons1, self.neurons2)
        self.fc3   = nn.Linear(self.neurons2, n_classes)
        self.last = self.fc3
        self.gradcam = self.conv2
        

    def forward(self, x, return_features = False):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        f = self.fc2(x)
        x = F.relu(f)
        x = self.fc3(x)
        if return_features:
            return x, f
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)
    
    def resize_conv1_layer(self, new_filters):
        if new_filters < self.filter1:
            new_conv = nn.Conv2d(1, self.filter1, 5, padding=2)
            new_conv.weight = self.conv1.weight[:new_filters, :, :, :]
            new_conv.bias = self.conv1.bias[:new_filters]
            if self.conv1.weight.grad is not None:
                new_conv.requires_grad = True
                new_conv.weight.grad = self.conv1.weight.grad[:new_filters, :, :, :]
                new_conv.bias.grad = self.conv1.bias.grad[:new_filters]

            next_conv = nn.Conv2d(new_filters, self.filter2, 5)
            next_conv.weight = self.conv2.weight[:, :new_filters, :, :]
            if self.conv2.weight.grad is not None:
                next_conv.requires_grad = True
                next_conv.weight.grad = self.conv2.weight.grad[:, :new_filters, :, :]

        elif new_filters > self.filter1:
            new_conv = nn.Conv2d(1, self.filter1, 5, padding=2)
            new_conv.weight[:self.filter1, :, :, :] = self.conv1.weight
            new_conv.bias[:self.filter1] = self.conv1.bias
            if self.conv1.weight.grad is not None:
                new_conv.requires_grad = True
                new_conv.weight.grad[:self.filter1, :, :, :] = self.conv1.weight.grad
                new_conv.bias.grad[:self.filter1] = self.conv1.bias.grad

            next = nn.Conv2d(new_filters, self.filter2, 5)
            next.weight[:, :self.filter1, :, :] = self.conv2.weight
            if self.conv2.weight.grad is not None:
                next.requires_grad = True
                next.weight.grad[:, :self.filter1, :, :] = self.conv2.weight.grad

        self.filter1 = new_filters
        self.conv1 = new_conv
        self.conv2 = next

    def resize_conv2_layer(self, new_filters):
        if new_filters < self.filter2:
            new_conv = nn.Conv2d(self.filter1, new_filters, 5)
            new_conv.weight = self.conv2.weight[:new_filters, :, :, :]
            new_conv.bias = self.conv2.bias[:new_filters]
            if self.conv2.weight.grad is not None:
                new_conv.requires_grad = True
                new_conv.weight.grad = self.conv2.weight.grad[:new_filters, :, :, :]
                new_conv.bias.grad = self.conv2.bias.grad[:new_filters]

            next = nn.Linear(new_filters*5*5, self.neurons1)
            next.weight = self.fc1.weight[:, :new_filters*5*5]
            if self.fc1.weight.grad is not None:
                next.requires_grad = True
                next.weight.grad = self.fc1.weight.grad[:, :new_filters*5*5]

        elif new_filters > self.filter2:
            new_conv = nn.Conv2d(self.filter1, new_filters, 5)
            new_conv.weight[:self.filter2, :, :, :] = self.conv2.weight
            new_conv.bias[:self.filter2] = self.conv2.bias
            if self.conv2.weight.grad is not None:
                new_conv.requires_grad = True
                new_conv.weight.grad[:self.filter2, :, :, :] = self.conv2.weight.grad
                new_conv.bias.grad[:self.filter2] = self.conv2.bias.grad

            next = nn.Linear(new_filters*5*5, self.neurons1)
            next.weight[:, :self.filter2, :, :] = self.fc1.weight
            if self.fc1.weight.grad is not None:
                next.requires_grad = True
                next.weight.grad[:, :self.filter2, :, :] = self.fc1.weight.grad

        self.filter1 = new_filters
        self.conv2 = new_conv
        self.fc1 = next

    def resize_fc1_layer(self, new_neurons):
        if new_neurons < self.neurons1:
            new_fc = nn.Linear(self.filter2*5*5, new_neurons)
            new_fc.weight = self.fc1.weight[:new_neurons, :]
            new_fc.bias = self.fc1.bias[:new_neurons]
            if self.fc1.weight.grad is not None:
                new_fc.requires_grad = True
                new_fc.weight.grad = self.fc1.weight.grad[:new_neurons, :]
                new_fc.bias.grad = self.fc1.bias.grad[:new_neurons]

            next = nn.Linear(new_neurons, self.neurons2)
            next.weight = self.fc2.weight[:, :new_neurons]
            if self.fc2.weight.grad is not None:
                next.requires_grad = True
                next.weight.grad = self.fc2.weight.grad[:, :new_neurons]

        elif new_neurons > self.neurons1:
            new_fc = nn.Linear(self.filter2*5*5, new_neurons)
            new_fc.weight[:self.neurons1, :] = self.fc1.weight
            new_fc.bias[:self.neurons1] = self.fc1.bias
            if self.fc1.weight.grad is not None:
                new_fc.requires_grad = True
                new_fc.weight.grad[:self.neurons1, :] = self.fc1.weight.grad
                new_fc.bias.grad[:self.neurons1] = self.fc1.bias.grad

            next = nn.Linear(new_neurons, self.neurons2)
            next.weight[:, :self.neurons1] = self.fc2.weight
            if self.fc2.weight.grad is not None:
                next.requires_grad = True
                next.weight.grad[:, :self.neurons1] = self.fc2.weight.grad

        self.neurons1 = new_neurons
        self.fc1 = new_fc
        self.fc2 = next

    def resize_fc2_layer(self, new_neurons):
        if new_neurons < self.neurons2:
            new_fc = nn.Linear(self.neurons1, new_neurons)
            new_fc.weight = self.fc2.weight[:new_neurons, :]
            new_fc.bias = self.fc2.bias[:new_neurons]
            if self.fc2.weight.grad is not None:
                new_fc.requires_grad = True
                new_fc.weight.grad = self.fc2.weight.grad[:new_neurons, :]
                new_fc.bias.grad = self.fc2.bias.grad[:new_neurons]

            next = nn.Linear(new_neurons, self.neurons3)
            next.weight = self.fc3.weight[:, :new_neurons]
            if self.fc3.weight.grad is not None:
                next.requires_grad = True
                next.weight.grad = self.fc3.weight.grad[:, :new_neurons]

        elif new_neurons > self.neurons2:
            new_fc = nn.Linear(self.neurons1, new_neurons)
            new_fc.weight[:self.neurons2, :] = self.fc2.weight
            new_fc.bias[:self.neurons2] = self.fc2.bias
            if self.fc2.weight.grad is not None:
                new_fc.requires_grad = True
                new_fc.weight.grad[:self.neurons2, :] = self.fc2.weight.grad
                new_fc.bias.grad[:self.neurons2] = self.fc2.bias.grad

            next = nn.Linear(new_neurons, self.neurons3)
            next.weight[:, :self.neurons2] = self.fc3.weight
            if self.fc3.weight.grad is not None:
                next.requires_grad = True
                next.weight.grad[:, :self.neurons2] = self.fc3.weight.grad

        self.neurons2 = new_neurons
        self.fc2 = new_fc
        self.fc3 = next

    def resize_fc3_layer(self, new_neurons):
        if new_neurons < self.neurons3:
            new_fc = nn.Linear(self.neurons2, new_neurons)
            new_fc.weight = self.fc3.weight[:new_neurons, :]
            new_fc.bias = self.fc3.bias[:new_neurons]
            if self.fc3.weight.grad is not None:
                new_fc.requires_grad = True
                new_fc.weight.grad = self.fc3.weight.grad[:new_neurons, :]
                new_fc.bias.grad = self.fc3.bias.grad[:new_neurons]

        elif new_neurons > self.neurons3:
            new_fc = nn.Linear(self.neurons2, new_neurons)
            new_fc.weight[:self.neurons3, :] = self.fc3.weight
            new_fc.bias[:self.neurons3] = self.fc3.bias
            if self.fc3.weight.grad is not None:
                new_fc.requires_grad = True
                new_fc.weight.grad[:self.neurons3, :] = self.fc3.weight.grad
                new_fc.bias.grad[:self.neurons3] = self.fc3.bias.grad

        self.neurons3 = new_neurons
        self.fc3 = new_fc
            

def lenet_mnist(dynamic=False):
    if dynamic:
        return LeNetDynamic(n_classes=10)
    return LeNet(n_classes=10)
