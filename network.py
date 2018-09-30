import torch
import torch.nn as nn


class Network(nn.Module):
  def __init__(self, num_classes = 200, image_size = 127, image_type='color'):
    super(Network,self).__init__()
    self.num_classes = num_classes
    self.input_size = 3
    self.feature_width = int(image_size/8)
    if image_type == 'gray':
       self.input_size = 1
    self.features = nn.Sequential(
         nn.Conv2d(self.input_size,64,11,stride=2),
         nn.BatchNorm2d(64),
         nn.ReLU(True),
         nn.MaxPool2d(kernel_size=3,stride=2),
         nn.Conv2d(64,192,kernel_size=5),
         nn.BatchNorm2d(192),
         nn.ReLU(True),
         nn.MaxPool2d(3,2),
         nn.Conv2d(192,384,kernel_size=3),
         nn.BatchNorm2d(384),
         nn.ReLU(True),
         nn.Conv2d(384,256,kernel_size=3),
         nn.BatchNorm2d(256),
         nn.ReLU(True),
         nn.Conv2d(256,256,kernel_size=3),
         nn.BatchNorm2d(256)
         )
    self.fc1 = nn.Linear(6*6*256,1024)
    self.fc2 = nn.Linear(1024,512)
    self.fc3 = nn.Linear(512,self.num_classes)
  
  def load_model(self,pretrain='/export/home/zby/facerecognition/data/imagenet_weights/alexnet-owt-4df8aa71.pth'):
    model_dict = self.state_dict()
    print('Loading model...')
    pretrain_dict = torch.load(pretrain)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k,v in pretrain_dict.items():
      if '0' in k and '1' not in k:
        new_state_dict[k] = v
      if '3' in k:
        new_state_dict[k.replace('3','4')] = v
      if '6' in k and 'clas' not in k:
        new_state_dict[k.replace('6','8')] = v
      if '8' in k:
        new_state_dict[k.replace('8','11')] = v
      if '10' in k:
        new_state_dict[k.replace('10','14')] = v
    print('Model has been loaded')
    model_dict.update(new_state_dict)
    self.load_state_dict(model_dict)

  def forward(self,image):
   # print(image.shape)
    feature = self.features(image)
    #print(feature.shape)
    feature = feature.view(-1,6*6*256)
    #print(feature.shape)
    out = self.fc1(feature)
    out = self.fc2(out)
    out = self.fc3(out)
    return out

if __name__ == '__main__':
  net = Network()

