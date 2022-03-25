# -*- coding: utf-8 -*-

import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import logging

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=False,
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128,
                                          shuffle=False)


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 48
        self.C_1 = 48
        self.p = 6
        self.conv1 = nn.Conv2d(3, self.C_1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.C_1)
        self.layer1 = self._make_layer(block, self.C_1, layers[0], 1)
        self.layer2 = self._make_layer(block, self.C_1 * 2, layers[1], 2)
        self.layer3 = self._make_layer(block, self.C_1 * 4, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(self.p)
        self.linear = nn.Linear(self.C_1 * 4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


model = ResNet(BasicBlock, [2, 2, 2]).to(device)

# add a logger
start_time = str(int(time.time()))
logger_name = 'log/mini_project_' + start_time + '.log'
logging.basicConfig(filename=logger_name, level=logging.DEBUG)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.debug('number of params are %s', count_parameters(model))
print('number of params are ', count_parameters(model))

# Hyper-parameters
num_epochs = 100
learning_rate = 0.01
decay_rate = 0
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO: add decay rate here
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0001)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Train the model
total_step = len(train_loader)
curr_lr = learning_rate

#recorder
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

for epoch in range(num_epochs):
    start = time.time()
    train_loss, train_acc, test_loss, test_acc = 0.0, 0.0, 0.0, 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += torch.max(outputs, dim=1)[1].eq(labels).sum()/len(labels)*100.0
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % 100 == 0:
        #     print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
        #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += torch.max(outputs, dim=1)[1].eq(labels).sum() / len(labels) * 100.0

            # if (i + 1) % 100 == 0:
            #     print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
            #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Decay learning rate
    if (epoch + 1) % 20 == 0:  # was 20
        # curr_lr *= 0.9  # TODO: was 3
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
    stop = time.time()
    train_loss_hist.append(train_loss/len(train_loader))
    train_acc_hist.append(train_acc/len(train_loader))
    test_loss_hist.append(test_loss/len(test_loader))
    test_acc_hist.append(test_acc/len(test_loader))
    msg = ('epoch {0}, train_loss {1:.4f}, train_accuracy {2:.4f}, '
           'test_loss {3:.4f}, test_accuracy {4:.4f}, cost time {5:.4f}'.format(epoch,
                                                             train_loss/len(train_loader),
                                                             train_acc/len(train_loader),
                                                             test_loss/len(test_loader),
                                                             test_acc/len(test_loader),
                                                             stop-start
                                                             ))
    logging.debug(msg)
    print('epoch {0}, train_loss {1:.4f}, train_accuracy {2:.4f}, '
          'test_loss {3:.4f}, test_accuracy {4:.4f}, cost time {5:.4f}'.format(epoch,
                                                            train_loss/len(train_loader),
                                                            train_acc/len(train_loader),
                                                            test_loss/len(test_loader),
                                                            test_acc/len(test_loader),
                                                            stop-start
                                                            ))
# Specify a path
PATH = 'project1_model_' + start_time + '.pt'

# Save
torch.save(model, PATH)

# # Load
#
#
model = torch.load(PATH)
logging.debug(model.eval())
#
# # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# data visualization
logging.debug('train_loss_hist: %s ', train_loss_hist)
logging.debug('train_acc_hist: %s', train_acc_hist)
logging.debug('test_loss_hist: %s', test_loss_hist)
logging.debug('test_acc_hist: %s', test_acc_hist)
