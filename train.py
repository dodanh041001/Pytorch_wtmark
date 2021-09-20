from unet import UNet
from vgg16 import VGG16

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.optim as optim
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg = VGG16()
        self.unet = UNet()

    def forward(self, x, y):
        """
        x is the watermarked image, y is the ground truth 
        """
        y_hat = self.unet(x)
        f_y = self.vgg(y)
        f_y_hat = self.vgg(y_hat)
        return y_hat, f_y, f_y_hat


def calc_loss(y_hat, y, f_y_hat, f_y):
    l1_loss = nn.L1Loss()(y_hat, y)
    l2_loss = nn.MSELoss()(f_y_hat, f_y)
    loss = l1_loss + (1.0/(128 * 112 * 112)) * l2_loss * l2_loss
    return loss


# def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, train_gtruth, val_gtruth):
#     for epoch in range(1, n_epochs + 1):
#         loss_train = 0.0
#         val_loss = 0.0
#         model.train()
#         for imgs in train_loader:

#             y_hat, f_y, f_y_hat = model(imgs[0], imgs[1])
#             loss_train += loss_fn(y_hat, imgs, f_y_hat, f_y)

        #     optimizer.zero_grad()  
        #     loss.backward() 
        #     optimizer.step()

        # model.eval()
        # for imgs in val_loader:
        #     y_hat, f_y, f_y_hat = model(imgs[0], imgs[1])
        #     val_loss += loss_fn(y_hat, imgs, f_y_hat, f_y)
            
        
#         if epoch == 1 or epoch % 1 == 0:
#             print('Epoch {}, Training loss {}, Validation loss {}'.format(
#                 epoch,
#                 loss_train / len(train_loader),
#                 val_loss / len(val_loader)))

def training_loop(n_epochs, optimizer, model, train_loader, val_loader, train_gtruth, val_gtruth):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        val_loss = 0.0
        model.train()
        for imgs, masks in zip(train_loader, train_gtruth):

            y_hat, f_y, f_y_hat = model(imgs, masks)
            loss = calc_loss(y_hat, imgs, f_y_hat, f_y)

            optimizer.zero_grad()  
            loss.backward() 
            optimizer.step()

        model.eval()
        for imgs, masks in zip(val_loader, val_gtruth):
            y_hat, f_y, f_y_hat = model(imgs, masks)
            val_loss += calc_loss(y_hat, imgs, f_y_hat, f_y)
        
        if epoch == 1 or epoch % 1 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(
                epoch,
                loss_train / len(train_loader),
                val_loss / len(val_loader)))
# Make sure to concatenate the input image and the target image before training
# class WaterDataSet(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.list_files = os.listdir
    
#     def __len__(self):
#         return len(self.list_files)
    
#     def __getitem__(self, index):
#         img_file = self.list_files[index]
#         img_path = os.path.join(self.root_dir, img_file)
#         image = np.array(Image.open(img_path).convert("RGB"))
#         split_half = int(image.shape[1]/2)
#         input_image = image[:, :split_half, :]
#         target_image = image[:, split_half:, :]

#         return input_image, target_image


class WaterDataSet:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        # image = Image.open(img_path).convert("RGB")
        # image = cv2.imread(img_path)
        
        image = Image.open(img_path).convert("RGB")
        # transform Image into the numpy array
        image = np.asarray(image)
        # transform the numpy array into the tensor
        image = torchvision.transforms.ToTensor()(image)
        
        return image
        
model = Net()
train_loader = WaterDataSet('/disk_local/unet-vgg/data_rever/train/imgs')
train_gtruth = WaterDataSet('/disk_local/unet-vgg/data_rever/train/masks')
val_loader = WaterDataSet('/disk_local/unet-vgg/data_rever/test/imgs')
val_gtruth = WaterDataSet('/disk_local/unet-vgg/data_rever/test/masks')

optimizer = optim.Adam(model.parameters(), lr=0.05)

training_loop(2, optimizer, model, train_loader, val_loader, train_gtruth, val_gtruth)
