import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init as torch_init

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.nc=3

        self.ngf=128

        self.ndf=128

        self.e1 = nn.Conv2d(self.nc, self.ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(self.ndf)
        torch_init.xavier_normal_(self.e1.weight)

        self.e2 = nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.ndf*2)
        torch_init.xavier_normal_(self.e2.weight)

        self.e3 = nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.ndf*4)
        torch_init.xavier_normal_(self.e3.weight)

        self.e4 = nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.ndf*8)
        torch_init.xavier_normal_(self.e4.weight)

        self.e5 = nn.Conv2d(self.ndf*8, self.ndf*8, 4, 2, 1)        
        self.bn5 = nn.BatchNorm2d(self.ndf*8)
        torch_init.xavier_normal_(self.e5.weight)

        self.fc1 = nn.Linear(4096, 100)
        torch_init.xavier_normal_(self.fc1.weight)        

        self.deconv1_1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv1_2 = nn.ConvTranspose2d(2, d*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2) 
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):        
        x = F.relu(self.bn1(self.e1(input)))        
        x = F.relu(self.bn2(self.e2(x)))        
        x = F.relu(self.bn3(self.e3(x)))        
        x = F.relu(self.bn4(self.e4(x)))    
        x = F.relu(self.bn5(self.e5(x)))    
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])        
        x = F.relu(self.fc1(x))
        x = x.view(-1,100,1,1)        
        x = F.leaky_relu(self.deconv1_1_bn(self.deconv1_1(x)), 0.2)
        y = F.leaky_relu(self.deconv1_2_bn(self.deconv1_2(label)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        # x = F.tanh(self.deconv4(x))
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(3, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(2, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        # x = F.sigmoid(self.conv4(x))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
