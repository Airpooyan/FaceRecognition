import torch
import torch.nn as nn

model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
          }

class BottleNeck(nn.Module):

    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        if stride !=1 or self.expansion*planes != inplanes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(self.expansion*planes)
                                            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(True)

    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        
        out = self.relu2(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        if self.downsample != None:
            residual = self.downsample(x)
        else:
            residual = x
        out = out+residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, num_blocks, input_size=128, num_classes=200):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        for p in self.parameters():
            p.requires_grad = False
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3], 512, stride=2)
        self.average = nn.AvgPool2d(int(input_size/32))
        self.fc = nn.Linear(512*BottleNeck.expansion,num_classes)

    def load_model(self,pretrain='/export/home/zby/facerecognition/data/imagenet_weights/resnet50-19c8e357.pth'):
        print('Loading pretrain model......')
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        from collections import OrderedDict
        new_dict = OrderedDict()
    #    for k in model_dict:
    #        print(k)
        for k,v in pretrain_dict.items():
            if 'fc' not in k:
                new_dict[k]=v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for now_stride in strides:
            layers.append(BottleNeck(self.inplanes, planes, stride=now_stride))
            self.inplanes = planes*BottleNeck.expansion
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        #print(out.shape)

        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)

    #    print(out.shape)
        out = self.average(out)
        out = self.fc(out.view(out.shape[0],-1))

        return out



def res50(pretrain=True):
    resnet = ResNet([3,4,6,3])
    if pretrain:
        resnet.load_model(pretrain='/export/home/zby/facerecognition/data/imagenet_weights/resnet50-19c8e357.pth')
    return resnet

def res101(pretrain=True):
    resnet = ResNet([3,4,23,3])
    if pretrain:
        resnet.load_model(pretrain='/export/home/zby/facerecognition/data/imagenet_weights/resnet101-5d3b4d8f.pth')
    return resnet

if __name__ == '__main__':
    resnet = res101()
    state_dict = resnet.state_dict()
    for k in state_dict:
        print(k)





