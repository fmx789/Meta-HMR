import math
import os
from collections import OrderedDict

from models.maml import MetaModule, MetaLinear, MetaConv2d, MetaBatchNorm2d, MetaSequential

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os.path as osp
from .hrnet import hrnet_w32, hrnet_w48
from models.backbones.hrnet.cls_hrnet import HighResolutionNet
from models.backbones.hrnet.hrnet_config import cfg
from models.backbones.hrnet.hrnet_config import update_config
from utils.geometry import rot6d_to_rotmat


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_meta(MetaModule):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_meta, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, params=None):
        residual = x

        out = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        out = self.bn1(out, params=self.get_subdict(params, 'bn1'))
        out = self.relu(out)

        out = self.conv2(out, params=self.get_subdict(params, 'conv2'))
        out = self.bn2(out, params=self.get_subdict(params, 'bn2'))
        out = self.relu(out)

        out = self.conv3(out, params=self.get_subdict(params, 'conv3'))
        out = self.bn3(out, params=self.get_subdict(params, 'bn3'))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CliffAuxNet_resnet(nn.Module):
    def __init__(self, block, layers, smpl_mean_params, bbox_type='square'):
        super(CliffAuxNet_resnet, self).__init__()
        self.inplanes = 64
        npose = 24 * 6
        nbbox = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if bbox_type == 'square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type == 'rect':
            self.avgpool = nn.AvgPool2d((8, 6), stride=1)

        self.fc1 = nn.Linear(512 * 4 + npose + nbbox + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]  # feature b,2048
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        # print()
        for i in range(n_iter):
            xc = torch.cat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class HmrAuxNet_resnet(nn.Module):
    def __init__(self, block, layers, smpl_mean_params, bbox_type='square'):
        super(HmrAuxNet_resnet, self).__init__()
        self.inplanes = 64
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if bbox_type == 'square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type == 'rect':
            self.avgpool = nn.AvgPool2d((8, 6), stride=1)

        self.fc1 = nn.Linear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]  # feature b,2048
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        # print()
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class CliffAuxNet_hrnet(nn.Module):
    """ SMPL Iterative Regressor with HRNet backbone
    """

    def __init__(self, smpl_mean_params, bbox_type='rect', img_feat_num=2048):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        super(CliffAuxNet_hrnet, self).__init__()
        curr_dir = osp.dirname(osp.abspath(__file__))
        config_file = osp.join(curr_dir, "./backbones/hrnet/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        update_config(cfg, config_file)
        self.encoder = HighResolutionNet(cfg)

        npose = 24 * 6
        nshape = 10
        ncam = 3
        nbbox = 3

        fc1_feat_num = 1024
        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        reg_in_feat_num = img_feat_num + nbbox + npose + nshape + ncam
        self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.encoder(x)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        print()
        for i in range(n_iter):
            xc = torch.cat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class CliffMetaNet_resnet(MetaModule):
    def __init__(self, block, layers, smpl_mean_params, bbox_type='square'):
        super(CliffMetaNet_resnet, self).__init__()
        self.inplanes = 64
        npose = 24 * 6
        nbbox = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if bbox_type == 'square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type == 'rect':
            self.avgpool = nn.AvgPool2d((8, 6), stride=1)
        self.fc1 = MetaLinear(512 * 4 + npose + nbbox + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = MetaLinear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = MetaLinear(1024, npose)
        self.decshape = MetaLinear(1024, 10)
        self.deccam = MetaLinear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # TODO n_iter
    def forward(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=10, params=None):

        batch_size = x.shape[0]  # feature b,2048
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0),-1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        # print()
        for i in range(n_iter):
            xc = torch.cat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc, params=self.get_subdict(params, 'fc1'))
            xc = self.drop1(xc)
            xc = self.fc2(xc, params=self.get_subdict(params, 'fc2'))
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc, params=self.get_subdict(params, 'decpose')) + pred_pose
            pred_shape = self.decshape(xc, params=self.get_subdict(params, 'decshape')) + pred_shape
            pred_cam = self.deccam(xc, params=self.get_subdict(params, 'deccam')) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class HmrMetaNet(MetaModule):
    def __init__(self, block, layers, smpl_mean_params, bbox_type='square'):
        super(HmrMetaNet, self).__init__()
        self.inplanes = 64
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if bbox_type == 'square':
            self.avgpool = nn.AvgPool2d(7, stride=1)
        elif bbox_type == 'rect':
            self.avgpool = nn.AvgPool2d((8, 6), stride=1)
        self.fc1 = MetaLinear(512 * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = MetaLinear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = MetaLinear(1024, npose)
        self.decshape = MetaLinear(1024, 10)
        self.deccam = MetaLinear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # TODO n_iter
    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=10, params=None):

        batch_size = x.shape[0]  # feature b,2048
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        # print()
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc, params=self.get_subdict(params, 'fc1'))
            xc = self.drop1(xc)
            xc = self.fc2(xc, params=self.get_subdict(params, 'fc2'))
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc, params=self.get_subdict(params, 'decpose')) + pred_pose
            pred_shape = self.decshape(xc, params=self.get_subdict(params, 'decshape')) + pred_shape
            pred_cam = self.deccam(xc, params=self.get_subdict(params, 'deccam')) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class CliffMetaNet_hrnet(MetaModule):
    """ SMPL Iterative Regressor with HRNet backbone
    """

    def __init__(self, smpl_mean_params, bbox_type='rect'):
        print('Model: Using {} bboxes as input'.format(bbox_type))
        super(CliffMetaNet_hrnet, self).__init__()
        self.inplanes = 64
        npose = 24 * 6
        nbbox = 3
        self.backbone_out = 720  # TODO: w32 480 w48 720
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = MetaLinear(self.backbone_out + npose + nbbox + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = MetaLinear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = MetaLinear(1024, npose)
        self.decshape = MetaLinear(1024, 10)
        self.deccam = MetaLinear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.backbone = hrnet_w48(pretrained_ckpt_path='models/backbones/pose_hrnet_w48_256x192.pth', downsample=True,
                                  use_conv=True)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, bbox_info, init_pose=None, init_shape=None, init_cam=None, n_iter=10, params=None):

        batch_size = x.shape[0]
        # print(x.shape)

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.backbone(x)
        xf = self.avgpool(xf)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        # print()
        for i in range(n_iter):
            xc = torch.cat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)
            # print('HMR')
            # print(xc.shape, xf.shape, bbox_info.shape)
            # xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc, params=self.get_subdict(params, 'fc1'))
            xc = self.drop1(xc)
            xc = self.fc2(xc, params=self.get_subdict(params, 'fc2'))
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc, params=self.get_subdict(params, 'decpose')) + pred_pose
            pred_shape = self.decshape(xc, params=self.get_subdict(params, 'decshape')) + pred_shape
            pred_cam = self.deccam(xc, params=self.get_subdict(params, 'deccam')) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


def build_model(smpl_mean_params, pretrained=True, backbone='resnet', name='cliff_meta', **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if backbone == 'resnet':
        if name == 'cliff_meta':
            model = CliffMetaNet_resnet(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
        elif name == 'cliff_aux':
            model = CliffAuxNet_resnet(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
        elif name == 'hmr_aux':
            model = HmrAuxNet_resnet(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
        elif name == 'hmr_meta':
            model = HmrMetaNet(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
    else:
        if name == 'cliff_meta':
            model = CliffMetaNet_hrnet(smpl_mean_params, **kwargs)
        elif name == 'cliff_aux':
            model = CliffAuxNet_hrnet(smpl_mean_params, **kwargs)
    if pretrained:
        if backbone == 'resnet':
            if name == 'cliff_meta' or name == 'hmr_meta':
                resnet_coco = torch.load(os.path.realpath('./models/backbones/pose_resnet.pth'))
                old_dict = resnet_coco['state_dict']
                new_dict = OrderedDict([(k.replace('backbone.', ''), v) for k, v in old_dict.items()])
                model.load_state_dict(new_dict, strict=False)
            elif name == 'cliff_aux':
                aux_net = torch.load(os.path.realpath('./models/ckpt/res50-PA45.7_MJE72.0_MVE85.3_3dpw.pt'))['model']
                new_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in aux_net.items()])
                new_dict = OrderedDict([(k.replace('encoder.', ''), v) for k, v in new_dict.items()])
                model.load_state_dict(new_dict, strict=True)
            elif name == 'hmr_aux':
                aux_net = torch.load(os.path.realpath('./models/ckpt/hmr.pt'))['model']
                model.load_state_dict(aux_net, strict=False)  # has not init pose/shape/cam
        else:
            if name == 'cliff_meta':
                pass
            elif name == 'cliff_aux':
                aux_net = torch.load(os.path.realpath('./models/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt'))['model']
                new_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in aux_net.items()])
                model.load_state_dict(new_dict, strict=True)
    return model
