import torch
import torch.nn as nn
from torch.nn import functional as F

class Loss(nn.Module):
  def  __init__(self):
    super(Loss,self).__init__()
    #self.loss= nn.CrossEntropyLoss()

  def forward(self,output,label):
    loss = F.cross_entropy(output,label)
    return loss
