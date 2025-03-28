import os
import time
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np
import scipy.io as sio
import scipy.ndimage
import glob
from PIL import Image
from torch.utils import data
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import numbers


## ---------------------- tool function ---------------------- ##
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
def stain_vector_decomposition(img):
    # img: [T+1, 3, H, W]
    sv = np.array([
                [0.5626, 0.2159],
                [0.7201, 0.8012],
                [0.4062, 0.5581]
                ])
    I0 = 240 / 255
    # reshape image
    img_v = img.transpose([0,2,3,1]).reshape((-1, 3))  # [T*H*W, 3]
    img_v = -np.log((img_v + 1/255)/I0)
    # project image onto stain vectors
    img_c = img_v.dot(np.linalg.pinv(sv).T)
    # project back to RGB space
    img_c = img_c.reshape([img.shape[0], img.shape[2], img.shape[3], 2]).astype('float32')
    # commbine two dimensions into one
    img = img_c[...,0:1] - img_c[...,1:2]
    return img.transpose([0,3,1,2])  # [T+1, 1, H, W]


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(pic, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            pic (np array): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to the crop for center crop.
        """

        h, w = pic.shape[-2:]
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

    def __call__(self, pic):
        """
        Args:
            pic (np array): Image to be cropped. # [C, H, W] or [T, C, H, W]
        Returns:
            np array: Cropped image.
        """

        # check type of [pic]
        if not isinstance(pic, np.ndarray):
            raise TypeError('img should be numpy array. Got {}'.format(type(pic)))

        # if image has only 2 channels make them 3
        if len(pic.shape) < 3:
            pic = pic.reshape(pic.shape[0], pic.shape[1], -1)

        # get crop params: starting pixels and size of the crop
        i, j, h, w = self.get_params(pic, self.size)

        return pic[..., i:i + h, j:j + w]


class CCUQDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, pos_data_path, neg_data_path, frames, ch_per_img=1):
        "Initialization"
        self.pos_data_path = pos_data_path
        self.neg_data_path = neg_data_path
        self.flist, self.labels = [], []
        if pos_data_path is not None:
            self.flist.extend(glob.glob(os.path.join(pos_data_path, '*.mat')))
            self.labels.extend([1] * len(glob.glob(os.path.join(pos_data_path, '*.mat'))))
        if neg_data_path is not None:
            self.flist.extend(glob.glob(os.path.join(neg_data_path, '*.mat')))
            self.labels.extend([0] * len(glob.glob(os.path.join(neg_data_path, '*.mat'))))
        self.frames = frames
        self.ch_per_img = ch_per_img

        self.cache = False

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.flist)

    def __getitem__(self, index):
        """
        Generates one sample of data
        he_outputs: [T+1, 3, H, W], cyclic inference in HE domain,
        dapi_outputs: [T+1, 1, H, W], cyclic inference in DAPI domain, the first frame is measurement
        tissue_masks: [T+1, H, W], tissue mask from HE images
        nuclei_masks: [T+1, H, W], nuclei mask from HE images
        """
        # print("DONE", time.time())
        if self.cache:
            # f = os.path.splitext(os.path.basename(self.flist[index]))[0]
            # y = np.random.randint(2)
            # if y:
            #     x = np.load(os.path.join(self.cache_path, '{}_pos.npy'.format(f)))
            # else:
            #     x = np.load(os.path.join(self.cache_path, '{}_neg.npy'.format(f)))
            f = os.path.splitext(os.path.basename(self.flist[index]))[0]
            tmp = np.load(os.path.join(self.cache_path, '{}.npz'.format(f)))
            x, y = tmp['x'], tmp['y']
            # if self.ch_per_img == 3:  # expand to 3 channels 
            #     x = x.reshape((x.shape[0]//6, 6, x.shape[1], x.shape[2]))  # [T', C, H, W]
            #     x_dapi, x_tm, x_nm = x[:, 3:4, :, :], x[:, 4:5, :, :], x[:, 5:6, :, :]  # [T', 1, H, W]
            #     x_dapi, x_tm, x_nm = np.repeat(x_dapi, 3, axis=1), np.repeat(x_tm, 3, axis=1), np.repeat(x_nm, 3, axis=1)  # [T', 3, H, W]
            #     x = np.concatenate([x[:, :3, :, :], x_dapi, x_tm, x_nm], axis=1)  # [T', 12, H, W]
            #     x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))  # [T'*C, H, W]
            x = x.astype('float32')
            y = np.array([y]).astype('int64')
            return torch.from_numpy(x), torch.from_numpy(y)
        else:
            # Select sample
            f = self.flist[index]

            # Load data
            tmp = sio.loadmat(f)
            
            # try:
            if self.ch_per_img == 0:
                # baseline (HE only)
                tmp = tmp['he_outputs']  # [T+1, 3, H, W]
            elif self.ch_per_img == 1:
                # 4 channels
                tmp_he = tmp['he_outputs']  # [T+1, 3, H, W]
                tmp_he = stain_vector_decomposition(tmp_he)  # [T+1, 1, H, W]
                tmp = np.concatenate([tmp_he, tmp['dapi_outputs'], np.expand_dims(tmp['tissue_masks'], axis=1), np.expand_dims(tmp['nuclei_masks'], axis=1)], axis=1).astype('float32')  # [T+1, 4, H, W]
            elif self.ch_per_img == 3:
                # 3 channels
                tmp_he = tmp['he_outputs']  # [T+1, 3, H, W]
                tmp_dapi = np.repeat(tmp['dapi_outputs'], 3, axis=1)  # [T+1, 3, H, W]
                # tmp_tm = np.repeat(np.expand_dims(tmp['tissue_masks'], axis=1), 3, axis=1)  # [T+1, 3, H, W]
                # tmp_nm = np.repeat(np.expand_dims(tmp['nuclei_masks'], axis=1), 3, axis=1)  # [T+1, 3, H, W]
                # truncate to the same length
                max_t = min(tmp_he.shape[0], tmp_dapi.shape[0])
                # max_t = min(tmp_he.shape[0], tmp_dapi.shape[0], tmp_tm.shape[0], tmp_nm.shape[0])
                tmp_he, tmp_dapi = tmp_he[:max_t, :, :, :], tmp_dapi[:max_t, :, :, :]
                # tmp_he, tmp_dapi, tmp_tm, tmp_nm = tmp_he[:max_t, :, :, :], tmp_dapi[:max_t, :, :, :], tmp_tm[:max_t, :, :, :], tmp_nm[:max_t, :, :, :]
                tmp = np.concatenate([tmp_he, tmp_dapi], axis=1).astype('float32')  # [T+1, 6, H, W]
                # tmp = np.concatenate([tmp_he, tmp_dapi, tmp_tm, tmp_nm], axis=1).astype('float32')  # [T+1, 12, H, W]
            x = tmp[self.frames, :, :, :]  # [T', C, H, W]
            # # downsample to 224*224
            # x = scipy.ndimage.zoom(x, (1, 1, 224/x.shape[2], 224/x.shape[3]), order=1)  # [T', C, 224, 224]
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))  # [T'*C, H, W]
            y = np.array([self.labels[index]]).astype('int64')


            return torch.from_numpy(x), torch.from_numpy(y)
            # except:
            #     pass
        
    def cache_data(self, cache_path=None):
        # cache input image sequence into small cache files for faster read
        if cache_path is None:
            self.cache_path = os.path.join(os.path.dirname(self.pos_data_path), '3DCNN_cache')
        else:
            self.cache_path = cache_path
        os.makedirs(self.cache_path, exist_ok=True)
        self.cache = False

        for i in tqdm(range(len(self))):
            x, y = self.__getitem__(i)
            x, y = x.numpy().astype('float16'), y.numpy().astype('int8')
            np.savez(os.path.join(self.cache_path, '{}.npz'.format(os.path.splitext(os.path.basename(self.flist[i]))[0])), x=x, y=y)
            # if y.item():
            #     np.save(os.path.join(self.cache_path, '{}_pos.npy'.format(os.path.splitext(os.path.basename(self.flist[i]))[0])), x)
            # else:
            #     np.save(os.path.join(self.cache_path, '{}_neg.npy'.format(os.path.splitext(os.path.basename(self.flist[i]))[0])), x)
                # np.save(os.path.join(self.cache_path, '{}_label.npy'.format(os.path.splitext(os.path.basename(self.flist[i]))[0])), y)
        
        self.cache = True


## ---------------------- end of Dataloaders ---------------------- ##



## -------------------- model prediction ---------------------- ##
def Conv3d_final_prediction(model, device, loader, prob=False):
    model.eval()

    all_y_pred = []
    all_y = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            if prob:
                y_pred = F.softmax(output, dim=1)
            else:
                y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze(-1).numpy().tolist())
            all_y.extend(y.cpu().data.squeeze(-1).numpy().tolist())

    return np.array(all_y_pred), np.array(all_y)


## -------------------- end of model prediction ---------------------- ##



## ------------------------ 3D CNN module ---------------------- ##
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=50):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_rep(nn.Module):
    # default a ResNet-152
    def __init__(
        self,
        in_channels: int = 3,
        in_imgs: int = 4,
        num_rep: int = 5,
        num_classes: int = 2,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_imgs = in_imgs
        self.num_rep = num_rep
        self.num_classes = num_classes

        # ResNet-152 backbone
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[0:-1])  # [N, 2048]
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.merge = nn.Conv1d(in_imgs * num_rep, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.merge_act = nn.ReLU()
        # self.merge = nn.Sequential(
        #     nn.Conv1d(in_imgs * num_rep, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        # )
        # self.merge = nn.Sequential(
        #     nn.Conv1d(in_imgs * num_rep, 64, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.ReLU(),
        # )
        self.fc1 = nn.Linear(2048, 1024)
        self.fc_act1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # reshape: [N, C, H, W] -> [N, in_imgs*n_rep, 3, H, W] -> [N*in_imgs*n_rep, 3, H, W]
        n, c, h, w = x.shape
        assert c == self.in_channels * self.in_imgs * self.num_rep
        x = x.reshape((n, c//self.in_channels, self.in_channels, h, w)).reshape((n*c//self.in_channels, self.in_channels, h, w))
        x = self.backbone(x)  # [N*in_imgs*n_rep, 2048, 1, 1]
        x = self.merge_act(self.merge(x.reshape((n, self.num_rep*self.in_imgs, -1))).squeeze(1))
        # x = self.merge(x.reshape((n, self.num_rep*self.in_imgs, -1))).squeeze(1)  # [N, n_rep*in_imgs, 2048] -> [N, 1, 2048] -> [N, 2048]
        x = self.fc_act1(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


## --------------------- end of 3D CNN module ---------------- ##

