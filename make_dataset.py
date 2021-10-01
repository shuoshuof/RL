import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms,models,datasets
import os
import cv2
from PIL import Image
batch_size = 512
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader,TensorDataset
# data_transforms={
#     'train':transforms.Compose(
#         [
#             transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),
#             transforms.RandomGrayscale(p=0.025),
#             transforms.ToTensor()
#         ]
#     ),
#     'valid':transforms.Compose(
#         [
#             transforms.ToTensor()
#         ]
#     )
# }
data_dir = './dataset2'
# image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle = True) for x in ['train','valid']}
# dataset_sizes = {x:len(image_datasets[x]) for x in ['train','valid']}
# class_names = image_datasets['train'].classes
# img = cv2.imread("./dataset/train/0_0.08.jpg")
# print(img.shape)
train_img = []
train_angle = []
valid_img = []
valid_angle= []
for i in ['train','valid']:
    path = os.path.join(data_dir,i)
    for f in os.listdir(path):
        extension = os.path.splitext(f)[-1]
        if (extension == '.jpg'):
            name = os.path.splitext(f)[0]
            angle = float(name.split("_")[2])
            print(angle)
            img = cv2.imread(os.path.join(path, f))
            print(img.shape)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img =cv2.resize(img,(66,200))
            # img = Image.open(os.path.join(path, f))
            # img = img.resize((66,200))
            # img = img.transpose(2,0,1)

            # print(img.shape)
            if i =='train':
                train_angle.append(angle)
                train_img.append(img)
            else:
                valid_img.append(img)
                valid_angle.append(angle)
train_img=np.array(train_img)
train_angle = np.array(train_angle)
valid_img = np.array(valid_img)
valid_angle=np.array(valid_angle)
np.save("train_img",train_img)
np.save("train_angle",train_angle)
np.save("valid_img",valid_img)
np.save("valid_angle",valid_angle)
print(train_img.shape)
print(train_angle.shape)
print(valid_img.shape)
print(valid_angle.shape)
# train_img=torch.tensor(train_img)
# train_angle = torch.tensor(train_angle)
# train = Data.TensorDataset(train_img,train_angle)
# datas = DataLoader(train, batch_size=512, shuffle=True, drop_last=False, num_workers=0)
# print(datas)