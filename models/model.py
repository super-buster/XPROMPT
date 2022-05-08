import torch.nn as nn
from models.functions import ReverseLayerF
from torchsummary import summary

class DANNModel(nn.Module):
    def __init__(self,pretrain_model):
        self.pretrain_model=pretrain_model
        self.domain_classifier=nn.Sequential()
        self.domain_classifier.add_module('fc',nn.Linear(100,2))
        self.domain_classifier.add_module('softmax',nn.LogSoftmax(dim=1))

    def forward(self,x):
        outputs=self.pretrain_model(x)
        return outputs
        


if __name__=='__main__':
    pass
