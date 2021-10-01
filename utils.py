import os
import json
import pickle
import random

import matplotlib.pyplot as plt
def read_data(root):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    for i in ['train', 'valid']:
        path = os.path.join(root, i)
        for f in os.listdir(path):
            extension = os.path.splitext(f)[-1]
            if (extension == '.jpg'):
                img_path = os.path.join(path,f)
                name = os.path.splitext(f)[0]
                pixel_error = float(name.split("_")[2])
                img_label = pixel_error
                if i == 'train':
                    train_images_path.append(img_path)
                    train_images_label.append(img_label)
                else:
                    val_images_path.append(img_path)
                    val_images_label.append(img_label)
    return train_images_path,train_images_label,val_images_path,val_images_label
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)
    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)*255
            print(img)
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(str(label))
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()