import torch
from torch import nn
import torchvision
from torchvision import transforms, models, datasets
import copy
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
from my_dataset import MyDataSet
from utils import read_data,plot_data_loader_image


def acc_calculate(output, label):
    rate = abs((label - output) / label)
    acc = len(label) - rate.gt(0.1).sum()
    return acc


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    """
    Support function for model training.

    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    print("1111")
                    print(preds.device)
                    print(outputs.device)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("running_corrects=",running_corrects.device)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model1(model, criterion, optimizer, scheduler, device, num_epochs=25):
    """
    Support function for model training.

    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze(-1)
                    # if outputs.shape != labels.shape:
                    #     print(outputs.shape)
                    #     print(outputs.shape)
                    loss = criterion(outputs, labels)
                    # preds = outputs
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print(outputs.device, labels.to('cpu').device)
                running_corrects += acc_calculate(outputs, labels)
                # print(labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,best_acc

def get_net():
    net = nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=5, padding=0, stride=(2, 2)),
        nn.ELU(),
        # nn.Dropout(p=0.2),
        nn.Conv2d(24, 36, kernel_size=5, padding=0, stride=(2, 2)),
        nn.ELU(),
        # nn.Dropout(p=0.2),
        nn.Conv2d(36, 48, kernel_size=5, padding=0, stride=(2, 2)),
        nn.ELU(),
        # nn.Dropout(p=0.2),
        nn.Conv2d(48, 64, kernel_size=3, padding=0),
        nn.ELU(),
        # nn.Dropout(p=0.2),
        nn.Conv2d(64, 64, kernel_size=3, padding=0),
        nn.ELU(),
        # nn.Dropout(p=0.2),
        nn.Flatten(),
        nn.Linear(1280, 100),
        nn.ELU(),
        # nn.Dropout(p=0.5),
        nn.Linear(100, 50),
        nn.ELU(),
        # nn.Dropout(p=0.5),
        nn.Linear(50, 10),
        nn.ELU(),
        # nn.Dropout(p=0.5),
        nn.Linear(10, 1)

    )
    return net
X = torch.randn(1, 3, 66, 220)
net = get_net()
output = net(X)
print(output)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)

# train_img = np.load("train_img.npy")
# train_angle = np.load("train_angle.npy").reshape((-1, 1))
# valid_img = np.load("valid_img.npy")
# valid_angle = np.load("valid_angle.npy")
# train_img = torch.tensor(train_img, dtype=torch.float32) / 127.5 - 1
# train_angle = torch.tensor(train_angle, dtype=torch.float32)
# valid_img = torch.tensor(valid_img, dtype=torch.float32) / 127.5 - 1
# valid_angle = torch.tensor(valid_angle, dtype=torch.float32)
#
# train_datasets = Data.TensorDataset(train_img, train_angle)
# train_dataloaders = DataLoader(train_datasets, batch_size=512, shuffle=True, drop_last=False, num_workers=0)
# valid_datasets = Data.TensorDataset(train_img, train_angle)
# valid_dataloaders = DataLoader(valid_datasets, batch_size=512, shuffle=True, drop_last=False, num_workers=0)
# image_datasets = {'train': train_datasets, 'val': valid_datasets}
# dataloaders = {'train': train_dataloaders, 'val': valid_dataloaders}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root="D:\saeon\controllers\car_controller\dataset2"

train_images_path, train_images_label, val_images_path, val_images_label = read_data(root)

data_transform = {
    "train": transforms.Compose([
                                transforms.ToTensor()
                                ]),
    "val": transforms.Compose([
                               transforms.ToTensor()
                                ])}
train_dataset = MyDataSet(images_path=train_images_path,
                           images_label=train_images_label,
                           transform=data_transform["train"])
val_dataset = MyDataSet(images_path=val_images_path,
                           images_label=val_images_label,
                           transform=data_transform["val"])

batch_size = 100
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=train_dataset.collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=val_dataset.collate_fn)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
optimizer_fit = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fit, step_size=4, gamma=0.1)
# loss_fn = nn.SmoothL1Loss()
loss_fn = nn.MSELoss()
model,acc  = train_model1(net, loss_fn, optimizer_fit, scheduler=scheduler, device=device, num_epochs=20)
# train_model(net,loss_fn,optimizer_fit,scheduler=scheduler,num_epochs=7)

torch.save(model.state_dict(), "model_ {:4f}.pth".format(acc))
print("Saved PyTorch Model State to model.pth")