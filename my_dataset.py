from PIL import Image
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_label: list, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
# def read_data(root):
#     assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
#     train_images_path = []  # 存储训练集的所有图片路径
#     train_images_label = []  # 存储训练集图片对应索引信息
#     val_images_path = []  # 存储验证集的所有图片路径
#     val_images_label = []  # 存储验证集图片对应索引信息
#     for i in ['train', 'valid']:
#         path = os.path.join(root, i)
#         for f in os.listdir(path):
#             extension = os.path.splitext(f)[-1]
#             if (extension == '.jpg'):
#                 img_path = os.path.join(path,f)
#                 name = os.path.splitext(f)[0]
#                 pixel_error = float(name.split("_")[2])
#                 img_label = pixel_error
#                 if i == 'train':
#                     train_images_path.append(img_path)
#                     train_images_label.append(img_label)
#                 else:
#                     val_images_path.append(img_path)
#                     val_images_label.append(img_label)
#     return train_images_path,train_images_label,val_images_path,val_images_label
# def plot_data_loader_image(data_loader):
#     batch_size = data_loader.batch_size
#     plot_num = min(batch_size, 4)
#     for data in data_loader:
#         images, labels = data
#         for i in range(plot_num):
#             img = images[i].numpy().transpose(1, 2, 0)*255
#             print(img)
#             label = labels[i].item()
#             plt.subplot(1, plot_num, i + 1)
#             plt.xlabel(str(label))
#             plt.xticks([])  # 去掉x轴的刻度
#             plt.yticks([])  # 去掉y轴的刻度
#             plt.imshow(img.astype('uint8'))
#         plt.show()
# train_images_path, train_images_label, val_images_path, val_images_label = read_data(root)
# data_transform = {
#     "train": transforms.Compose([
#                                 transforms.ToTensor()
#                                 ]),
#     "val": transforms.Compose([
#                                transforms.ToTensor()
#                                 ])}
# train_data_set = MyDataSet(images_path=train_images_path,
#                            images_label=train_images_label,
#                            transform=data_transform["train"])
# batch_size = 8
# train_loader = torch.utils.data.DataLoader(train_data_set,
#                                            batch_size=batch_size,
#                                            shuffle=True,
#                                            num_workers=0,
#                                            collate_fn=train_data_set.collate_fn)
#
# plot_data_loader_image(train_loader)