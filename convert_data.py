import json,os
import random

if __name__ == '__main__':
  data_dir = 'data'
  all_classes = os.listdir(data_dir)
  all_data = []
  for now_class in all_classes:
      now_dir = os.path.join(data_dir,now_class)
      now_images = os.listdir(now_dir)
      for image in now_images:
          if image.endswith('.tif') or image.endswith('.jpg'):  
             data_dict = dict()
             data_dict['image_path'] = os.path.join(now_class,image)
             data_dict['image_label'] = int(now_class)
             all_data.append(data_dict)
  print(len(all_data))
  random.shuffle(all_data)
  train_data = all_data[0:1000]
  val_data = all_data[1000:]
  print(len(val_data))
  with open('train.json','w') as f:
    json.dump(train_data,f)
  with open('val.json','w') as f:
    json.dump(val_data,f)

