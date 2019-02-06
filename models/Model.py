from collections import OrderedDict
from models.Inception import inception_v3_without_last_layer
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vector_size, fix_weight):
        super(Model, self).__init__()
        self.inception = inception_v3_without_last_layer(pretrained=True)
        for param in self.inception.parameters():
            param.requires_grad = fix_weight
        self.fc = nn.Linear(
            in_features=2048,
            out_features=vector_size
        )
        init.normal(self.fc.weight, std=0.01)
        init.constant(self.fc.bias, 0)

    def forward(self, X):
        x = self.inception(X)
        x = self.fc(x)
        return x
