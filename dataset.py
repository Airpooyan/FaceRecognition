from torch.utils.data import Dataset,DataLoader
import json,os
import cv2
import torch
import torchvision.transforms as transforms

class FERET(Dataset):
  def __init__(self,num_classes=200,mode='train'):
    self.num_classes = num_classes
    self.data_dir = '/export/home/zby/facerecognition/data'
    self.image_dir = self.data_dir+'/images'
    if mode == 'train':
      self.anno_path = self.data_dir+'/annotations/train.json'
    else:
      self.anno_path = self.data_dir+'/annotations/val.json'
    print('Loading data ...')
    with open(self.anno_path,'r') as f:
      self.data = json.load(f)
    #crop = transforms.RandomResizeCrop(size=80,scale=(0.08,1.0))
    if mode == 'train':
       self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomResizedCrop(size=127,scale=(0.7,1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])])
    else:
       self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(155),
                                            transforms.CenterCrop(127),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])])
    print('Data has been loaded')

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    now_data = self.data[index]
    image_path = os.path.join(self.image_dir,now_data['image_path'])
    image_label = now_data['image_label'] - 1 #0-index
    im = cv2.imread(image_path)
    #print(im.shape,im.size)
    image = self.transform(im)#ToTensor将 H W C转换为 C H W格式，并归一化到0-1之间
    #print(image)  
    #print(image_label)
    return image,image_label

if __name__ == '__main__':
  data = FERET()
  dataLoader = DataLoader(data,batch_size = 2, shuffle = True, num_workers=1,pin_memory = True)
  for i,(image,label) in enumerate(dataLoader):
    label = label.long().cuda()
    print(label)
    #print(image.shape)
    #print(image)
    image = image.cuda()
    #print(image)
 #   print(label.shape)
    

