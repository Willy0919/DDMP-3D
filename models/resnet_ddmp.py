import torch.nn as nn
from lib.rpn_util import *
from models import resnet as resnet_mod
import torch
import numpy as np
from models.deform_conv_v2 import *
from maskrcnn_benchmark.layers import DeformUnfold
from maskrcnn_benchmark.layers import DeformConv
from maskrcnn_benchmark.modeling.make_layers import group_norm


class DynamicWeightsCat33(nn.Module):

    def __init__(self, channels, rgb_inchannel, group=1, kernel=3, dilation=(1, 1, 1), shuffle=False, deform=None):
        super(DynamicWeightsCat33, self).__init__()
        in_channel = channels
        # down the channels of input image features
        self.smooth_rgb = nn.Sequential(nn.Conv2d(rgb_inchannel, in_channel, 1, bias=False),
                                        nn.BatchNorm2d(in_channel),
                                        nn.ReLU(inplace=True))

        self.filter1 = Warp(in_channel, groups_k=group)
        self.filter2 = Warp(in_channel, groups_k=group)
        self.filter3 = Warp(in_channel, groups_k=group)

        if deform == 'deformatt':
            self.cata_off = nn.Conv2d(in_channel, 18, 3, padding=dilation[0],
                                      dilation=dilation[0], bias=False)
            self.catb_off = nn.Conv2d(in_channel, 18, 3, padding=dilation[1],
                                      dilation=dilation[1], bias=False)
            self.catc_off = nn.Conv2d(in_channel, 18, 3, padding=dilation[2],
                                      dilation=dilation[2], bias=False)

            # learn kernel

            self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=dilation[0], dilation=dilation[0])
            self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=dilation[1], dilation=dilation[1])
            self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=dilation[2], dilation=dilation[2])

        self.softmax = nn.Softmax(dim=-1)

        self.shuffle = shuffle
        self.deform = deform
        self.group = group
        self.K = kernel * kernel

        self.scale2 = nn.Sequential(nn.Conv2d(in_channel * 4, rgb_inchannel, 1, padding=0, bias=True),
                                    group_norm(rgb_inchannel),
                                    nn.ReLU(inplace=True))

    def forward(self, rgb_feat, depth1, depth2, depth3):
        # blur_depth = x
        rgb_org = rgb_feat

        x = self.smooth_rgb(rgb_feat)
        N, C, H, W = x.size()
        R = C // self.group
        affinity1, filter_w1 = self.filter1(depth1)
        affinity2, filter_w2 = self.filter2(depth2)
        affinity3, filter_w3 = self.filter3(depth3)
        if self.deform == 'deformatt':
            offset_1 = self.cata_off(x)
            offset_2 = self.catb_off(x)
            offset_3 = self.catc_off(x)

            offset1 = offset_1[:, :18, :, :]
            filter_w1 = filter_w1.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            filter_w1 = filter_w1.sigmoid()

            offset2 = offset_2[:, :18, :, :]
            filter_w2 = filter_w2.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            filter_w2 = filter_w2.sigmoid()

            offset3 = offset_3[:, : 18, :, :]
            filter_w3 = filter_w3.view(N, -1, H * W).view(N, 1, -1, H * W)  # N, 1, 9, H*W
            filter_w3 = filter_w3.sigmoid()

        if self.deform == 'none':
            xd_unfold1 = self.unfold1(x)
            xd_unfold2 = self.unfold2(x)
            xd_unfold3 = self.unfold3(x)
        else:
            xd_unfold1 = self.unfold1(x, offset1)
            xd_unfold2 = self.unfold2(x, offset2)
            xd_unfold3 = self.unfold3(x, offset3)

        if self.deform == 'deformatt':
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W)

            xd_unfold1 *= filter_w1
            xd_unfold2 *= filter_w2
            xd_unfold3 *= filter_w3

            xd_unfold1 = xd_unfold1.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1,
                                                                                                                   3, 2,
                                                                                                                   4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1,
                                                                                                                   3, 2,
                                                                                                                   4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.permute(0, 1, 3, 2).contiguous().view(N, self.group, R, H * W, self.K).permute(0, 1,
                                                                                                                   3, 2,
                                                                                                                   4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            #
        else:
            xd_unfold1 = xd_unfold1.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R,
                                                                                                    H * W,
                                                                                                    self.K).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    2,
                                                                                                                    4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold2 = xd_unfold2.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R,
                                                                                                    H * W,
                                                                                                    self.K).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    2,
                                                                                                                    4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)
            xd_unfold3 = xd_unfold3.view(N, C, self.K, H * W).permute(0, 1, 3, 2).contiguous().view(N, self.group, R,
                                                                                                    H * W,
                                                                                                    self.K).permute(0,
                                                                                                                    1,
                                                                                                                    3,
                                                                                                                    2,
                                                                                                                    4).contiguous().view(
                N * self.group * H * W, R, self.K)  # (BGHW, R, K)

        ## N, K, H, W --> N*H*W
        # use softmax or not
        # affinity1 = affinity1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
        #                                                                                             self.K)
        # affinity2 = affinity2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
        #                                                                                             self.K)
        # affinity3 = affinity3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
        #                                                                                            self.K)
        affinity1 = self.softmax(affinity1.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                                                                                                                self.K))
        affinity2 = self.softmax(affinity2.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
                                                                                                                self.K))
        affinity3 = self.softmax(affinity3.view(N, self.group, self.K, -1).permute(0, 1, 3, 2).contiguous().view(-1,
        self.K))

        out1 = torch.bmm(xd_unfold1, affinity1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(
            N, self.group * R, H, W)
        out2 = torch.bmm(xd_unfold2, affinity2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(
            N, self.group * R, H, W)
        out3 = torch.bmm(xd_unfold3, affinity3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, self.group, H * W, R).permute(0, 1, 3, 2).contiguous().view(N, self.group * R, H * W).view(
            N, self.group * R, H, W)

        out = self.scale2(torch.cat((x, out1, out2, out3), 1)) + rgb_org

        return out


class Warp(nn.Module):
    def __init__(self, channels, dilation=1, kernel=3, groups=1, groups_k=1, deform=False, normalize=None, att=False,
                 need_offset=False):
        super(Warp, self).__init__()
        self.group = groups
        self.need_offset = need_offset
        offset_groups = groups
        # if need_offset:
        self.off_conv = nn.Conv2d(channels, kernel * kernel * 2 * offset_groups + 9, 3,
                                  padding=dilation, dilation=dilation, bias=False)
        self.conv = DeformConv(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False,
                               deformable_groups=offset_groups)

        self.conv_1 = nn.Conv2d(channels, kernel * kernel * groups_k, kernel_size=3, padding=dilation,
                                dilation=dilation,
                                bias=False)
        self.bn = nn.BatchNorm2d(kernel * kernel * groups_k)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, H, W = x.size()

        R = C // self.group
        offset_filter = self.off_conv(x)
        offset = offset_filter[:, :18, :, :]
        filter = offset_filter[:, -9:, :, :]
        out = self.conv(x, offset)
        x = x + out
        x = self.relu(self.bn(self.conv_1(x)))
        return x, filter


class RPN(nn.Module):

    def __init__(self, phase, conf):
        super(RPN, self).__init__()

        self.base = resnet_mod.ResNetDilate(conf.base_model)
        self.adaptive_diated = conf.adaptive_diated
        self.dropout_position = conf.dropout_position
        self.use_dropout = conf.use_dropout
        self.drop_channel = conf.drop_channel
        self.use_corner = conf.use_corner
        self.corner_in_3d = conf.corner_in_3d
        self.deformable = conf.deformable

        if conf.use_rcnn_pretrain:
            # print(self.base.state_dict().keys())
            if conf.base_model == 101:
                pretrained_model = torch.load('faster_rcnn_1_10_14657.pth')['model']
                rename_dict = {'RCNN_top.0': 'layer4', 'RCNN_base.0': 'conv1', 'RCNN_base.1': 'bn1',
                               'RCNN_base.2': 'relu',
                               'RCNN_base.3': 'maxpool', 'RCNN_base.4': 'layer1',
                               'RCNN_base.5': 'layer2', 'RCNN_base.6': 'layer3'}
                change_dict = {}
                for item in pretrained_model.keys():
                    for rcnn_name in rename_dict.keys():
                        if rcnn_name in item:
                            change_dict[item] = item.replace(rcnn_name, rename_dict[rcnn_name])
                            break
                pretrained_model = {change_dict[k]: v for k, v in pretrained_model.items() if k in change_dict}
                self.base.load_state_dict(pretrained_model)

            elif conf.base_model == 50:
                pretrained_model = torch.load('data/res50_faster_rcnn_iter_1190000.pth',
                                              map_location=lambda storage, loc: storage)
                pretrained_model = {k.replace('resnet.', ''): v for k, v in pretrained_model.items() if 'resnet' in k}
                # print(pretrained_model.keys())
                self.base.load_state_dict(pretrained_model)

        self.depthnet = resnet_mod.ResNetDilate(50)
        out_channels = 2048

        dw_config = {}
        dw_group = dw_config.get('group', 4)
        dw_kernel = dw_config.get('kernel', 3)
        dw_dilation = dw_config.get('dilation', (1, 1, 1, 1))
        dw_shuffle = dw_config.get('shuffle', False)
        dw_deform = dw_config.get('deform', 'deformatt')
        self.dw_block1 = DynamicWeightsCat33(channels=256,
                                             rgb_inchannel=512,
                                             group=dw_group,
                                             kernel=dw_kernel,
                                             dilation=dw_dilation,
                                             shuffle=dw_shuffle,
                                             deform=dw_deform)
        self.dw_block2 = DynamicWeightsCat33(channels=256,
                                             rgb_inchannel=1024,
                                             group=dw_group,
                                             kernel=dw_kernel,
                                             dilation=dw_dilation,
                                             shuffle=dw_shuffle,
                                             deform=dw_deform)

        ##### scale the depth feature map and down the dimension
        ########## for the image stage2
        #####stage2 depth(dim=512), stage3 depth(upsample,dim=1024), stage4 depth(upsample,dim=2048)

        self.smooth_depth11 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth12 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth13 = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

        self.smooth_depth21 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth22 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))
        self.smooth_depth23 = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]

        self.prop_feats = nn.Sequential(
            nn.Conv2d(out_channels, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        if self.use_dropout:
            self.dropout = nn.Dropout(p=conf.dropout_rate)

        if self.drop_channel:
            self.dropout_channel = nn.Dropout2d(p=0.3)

        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1)

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # depth z regression
        self.prop_feats_dep = nn.Sequential(
            nn.Conv2d(out_channels, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        if self.use_dropout:
            self.dropout_dep = nn.Dropout(p=conf.dropout_rate)

        if self.drop_channel:
            self.dropout_channel_dep = nn.Dropout2d(p=0.3)

        ## predict z from depth map
        self.bbox_z3d_dep = nn.Conv2d(self.prop_feats_dep[0].out_channels, self.num_anchors, 1)
        self.bbox_x3d_dep = nn.Conv2d(self.prop_feats_dep[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d_dep = nn.Conv2d(self.prop_feats_dep[0].out_channels, self.num_anchors, 1)

        if self.corner_in_3d:
            self.bbox_3d_corners = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors * 18, 1)  # 2 * 8 + 2
            self.bbox_vertices = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors * 24, 1)  # 3 * 8
        elif self.use_corner:
            self.bbox_vertices = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors * 24, 1)

        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors

    def forward(self, x, depth):

        batch_size = x.size(0)

        x = self.base.conv1(x)
        depth = self.depthnet.conv1(depth)
        x = self.base.bn1(x)
        depth = self.depthnet.bn1(depth)
        x = self.base.relu(x)
        depth = self.depthnet.relu(depth)
        x = self.base.maxpool(x)
        depth = self.depthnet.maxpool(depth)

        x = self.base.layer1(x)
        depth = self.depthnet.layer1(depth)
        # x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

        x = self.base.layer2(x)
        depth1 = self.depthnet.layer2(depth)  # b, 512, 64,220
        depth2 = self.depthnet.layer3(depth1)  # b, 1024, 32, 110
        depth3 = self.depthnet.layer4(depth2)  # b, 2048, 32, 110

        if self.deformable:
            depth = self.deform_layer(depth1)
            x = x * depth1

        ### up sample depth4 and depth 3
        depth1_stage2 = self.smooth_depth11(depth1)
        depth2_stage2 = self.smooth_depth12(F.interpolate(depth2, scale_factor=2, mode='nearest'))
        depth3_stage2 = self.smooth_depth13(F.interpolate(depth3, scale_factor=2, mode='nearest'))

        x = self.dw_block1(x, depth1_stage2, depth2_stage2, depth3_stage2)

        if self.use_dropout and self.dropout_position == 'adaptive':
            x = self.dropout(x)

        if self.drop_channel:
            x = self.dropout_channel(x)

        x = self.base.layer3(x)

        ### down sample depth2
        depth1_stage3 = self.smooth_depth21(F.max_pool2d(depth1, 2))
        depth2_stage3 = self.smooth_depth22(depth2)
        depth3_stage3 = self.smooth_depth23(depth3)

        x = self.dw_block2(x, depth1_stage3, depth2_stage3, depth3_stage3)

        x = self.base.layer4(x)
        x = x * depth3

        if self.use_dropout and self.dropout_position == 'early':
            x = self.dropout(x)
            depth3 = self.dropout_dep(depth3)

        prop_feats = self.prop_feats(x)
        prop_feats_dep = self.prop_feats_dep(depth3)

        if self.use_dropout and self.dropout_position == 'late':
            prop_feats = self.dropout(prop_feats)
            prop_feats_dep = self.dropout(prop_feats_dep)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)
        # targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY

        # predict z from depth
        bbox_z3d_dep = self.bbox_z3d_dep(prop_feats_dep)
        bbox_x3d_dep = self.bbox_x3d_dep(prop_feats_dep)
        bbox_y3d_dep = self.bbox_y3d_dep(prop_feats_dep)

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        # although it's the same with x.view(batch_size, -1, 1) when c == 1, useful when c > 1
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d_dep = flatten_tensor(bbox_z3d_dep.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_x3d_dep = flatten_tensor(bbox_x3d_dep.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d_dep = flatten_tensor(bbox_y3d_dep.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)
        bbox_3d_dep = torch.cat((bbox_x3d_dep, bbox_y3d_dep, bbox_z3d_dep), dim=2)
        if self.corner_in_3d:
            corners_3d = self.bbox_3d_corners(prop_feats)
            corners_3d = flatten_tensor(corners_3d.view(batch_size, 18, feat_h * self.num_anchors, feat_w))
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))
        elif self.use_corner:
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            # print(cls.size(), prob.size(), bbox_2d.size(), bbox_3d.size(), feat_size)
            if self.corner_in_3d:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(
                    np.array(feat_size)).cuda(), bbox_vertices, corners_3d
            elif self.use_corner:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(np.array(feat_size)).cuda(), bbox_vertices
            else:
                return cls, prob, bbox_2d, bbox_3d, bbox_3d_dep, torch.from_numpy(np.array(feat_size)).cuda()

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = self.rois.type(torch.cuda.FloatTensor)

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):
    train = phase.lower() == 'train'

    rpn_net = RPN(phase, conf)
    print(rpn_net)
    if train:
        rpn_net.train()
    else:
        rpn_net.eval()

    return rpn_net


