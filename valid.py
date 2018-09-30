from network import Network
from loss import Loss
from dataset import FERET

import torch,os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Train FaceRecognition Network')

  parser.add_argument('--gpu',help='decide which gpu to use',
                       required = False,
                       default = '0',
                       type = str)
  parser.add_argument('--b',help='decide the batchsize',
                      required = False,
                      default = 1,
                      type = int)
  parser.add_argument('--e',help='decide epoch number',
                      required = False,
                      default = 100,
                      type = int)
  args = parser.parse_args()
  return args

def valid_model(dataLoader,epoch_number,model,criterion,batch_size):
  save_step = 10
  save_dir = 'output/models'
  with torch.no_grad():
    all_size,all_count,all_loss = 0,0,0
    for i,(image,label) in enumerate(dataLoader):
        image,label = image.cuda(), label.cuda()
        output = model(image)
        _,now_result = torch.max(output,1)
        all_size += label.shape[0]
        true_count = (now_result == label).sum().float()
        all_count += true_count
        now_accuracy = true_count/label.shape[0]
        now_accuracy = now_accuracy.data.item()*100
        loss = criterion(output,label)
        now_loss = loss.data.item()    
        all_loss += now_loss
    epoch_loss = all_loss/all_size
    epoch_accuracy = all_count/all_size
    pbar_str = 'Epoch:{}     Loss:{}   Epoch_Accuracy:{:.2f}%     '.format(epoch_number,now_loss,epoch_accuracy*100)
    print(pbar_str) 
    
  # print('Epoch Accuracy:{:.2f}%'.format(epoch_accuracy*100))

if __name__ == '__main__':
  args = parse_args()
  gpus = [int(i) for i in args.gpu.split(',')]
  epoch_number = args.e
  batch_size = args.b

  model = Network()
  model_dict = model.state_dict()
  now_model = os.path.join('output/models','epoch_{}.pth'.format(epoch_number))
  now_model = torch.load(now_model)
  from collections import OrderedDict
  now_dict = OrderedDict()
  for k,v in now_model.items():
      now_dict[k.replace('module.','')]=v
  model_dict.update(now_dict)
  model.load_state_dict(model_dict)
  model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
  model.eval()
  valid_set = FERET(mode='valid')
  print(len(valid_set))
  dataLoader = DataLoader(valid_set,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 1,
                          pin_memory = True)#pin_memory 意味着在return之前会转移到cuda中
 # print(len(dataLoader))
  criterion = Loss()
  valid_model(dataLoader,epoch_number,model,criterion,batch_size)
