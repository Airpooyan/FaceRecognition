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
                       default = '0,1',
                       type = str)
  parser.add_argument('--b',help='decide the batchsize',
                      required = False,
                      default = 16,
                      type = int)
  parser.add_argument('--e',help='decide epoch number',
                      required = False,
                      default = 100,
                      type = int)
  args = parser.parse_args()
  return args

def train_model(dataLoader,epoch_number,model,optimizer,criterion,batch_size):
  save_step = 10
  save_dir = 'output/models'
  for epoch in range(1,epoch_number+1):
    number_batch = len(dataLoader)
    pbar = tqdm(range(number_batch))
    all_size, all_count = 0,0
    for i,(image,label) in enumerate(dataLoader):
        image,label = image.cuda(), label.cuda()
        #print(image.shape)
        optimizer.zero_grad()
        output = model(image)
        _,now_result = torch.max(output,1)
        all_size += label.shape[0]
        true_count = (now_result == label).sum().float()
        all_count += true_count
        now_accuracy = true_count/label.shape[0]
        now_accuracy = now_accuracy.data.item()*100
        loss = criterion(output,label)
        loss.backward()
        now_loss = loss.data.item()
        pbar_str = 'Epoch:{}     Loss:{}   Accuracy:{}%     '.format(epoch,now_loss,now_accuracy)
        pbar.set_description(pbar_str)
        optimizer.step()
    epoch_accuracy = all_count/all_size
    pbar_str = 'Epoch:{}     Loss:{}   Epoch_Accuracy:{:.2f}%     '.format(epoch,now_loss,epoch_accuracy*100)
    pbar.set_description(pbar_str)
    if epoch % save_step == 0:
       if not os.path.exists(save_dir):
          os.makedirs(save_dir)
       out_dir = os.path.join(save_dir,'epoch_{}.pth'.format(epoch))
       torch.save(model.state_dict(),out_dir)
      
    
  # print('Epoch Accuracy:{:.2f}%'.format(epoch_accuracy*100))

if __name__ == '__main__':
  args = parse_args()
  gpus = [int(i) for i in args.gpu.split(',')]
  epoch_number = args.e
  batch_size = args.b

  model = Network()
  model.load_model()
  model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
  train_set = FERET()
  
  dataLoader = DataLoader(train_set,
                          batch_size = batch_size,
                          shuffle = True,
                          num_workers = 1,
                          pin_memory = True)#pin_memory 意味着在return之前会转移到cuda中
 # print(len(dataLoader))
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0.0005)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[70,85],gamma = 0.1)
  criterion = Loss()
  train_model(dataLoader,epoch_number,model,optimizer,criterion,batch_size)
