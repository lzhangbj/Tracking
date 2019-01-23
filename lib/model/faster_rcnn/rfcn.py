import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from subprocess import call
import gc
import sys
import torch.cuda as cutorch

def memReport():
    count = 0
    gc.enable()
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            # print(obj)
            count+=1
    print("tot {:d} gpu tensors".format(count))

def gpuStats():
    for i in range(cutorch.device_count()):
        sys.stdout.write("{}: {:d}/{:d}\n".format(cutorch.get_device_name(i), int(cutorch.memory_allocated(i)*1e-6), int(cutorch.max_memory_allocated(i)*1e-6)))



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class _rfcn(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_rfcn, self).__init__()
        self.k = 7
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RFCN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        # self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()

        # self.conv_new_1 = Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, relu=True)

        # self.rfcn_cls = nn.Conv2d(in_channels=1024, out_channels=self.n_classes * self.k * self.k,
        #                         kernel_size=1, stride=1, padding=0, bias=False)
        self.RFCN_cls_net = nn.Conv2d(512,self.n_classes*7*7, [1,1], padding=0, stride=1)
        self.psroipooling_cls = PSRoIPool(pooled_height=self.k, pooled_width=self.k,
                                        spatial_scale= 1/16.0, group_size=self.k, 
                                        output_dim=self.n_classes)
        # self.rfcn_bbox = nn.Conv2d(in_channels=1024, out_channels= self.n_classes*4 * self.k * self.k,
        #                         kernel_size=1, stride=1, padding=0, bias=False) #--class-agnostic
        if not self.class_agnostic:
            self.RFCN_bbox_net = nn.Conv2d(512, 4*self.n_classes*7*7, [1,1], padding=0, stride=1)
            self.psroipooling_loc = PSRoIPool(pooled_height=self.k, pooled_width=self.k,
                                            spatial_scale= 1/16.0,   group_size=self.k,
                                            output_dim= self.n_classes * 4) #----class-agnostic
        else:
            self.RFCN_bbox_net = nn.Conv2d(512, 4*7*7, [1,1], padding=0, stride=1)
            self.psroipooling_loc = PSRoIPool(pooled_height=self.k, pooled_width=self.k,
                                            spatial_scale= 1/16.0,   group_size=self.k,
                                            output_dim= 4) #----class-agnostic




        self.pooling = nn.AvgPool2d((7,7), stride=(7,7))

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RFCN_base(im_data) # (batch, 1024, h, w)
        # zl modified, now has 2048 channels
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RFCN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # if it is training phrase, then use ground truth bboxes for refining
        # sample boxes from both proposed and groundtruth boxes
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois) # remove batch become shape (num_rois, 5)
        # new_feat = self._head_to_tail(base_feat) # h and w halved (batch, 1024, h, w)
        # print(base_feat)
        # do roi pooling based on predicted rois
        # new_feat = self.conv_new_1(base_feat) # change 2048 channel to 1024 only (batch, 1024, h, w)
        rfcn_cls = self.RFCN_cls_net(base_feat) # output (batch, k*k*n_classes, h, w)
        rfcn_bbox = self.RFCN_bbox_net(base_feat)  # output (batch, 8*k*k, h, w) 
        #----cls
        psroipooled_cls_rois = self.psroipooling_cls(rfcn_cls, rois.view(-1, 5)) # shape a batchsize_length list of (num_rois,  n_classes, k, k)
        ave_cls_score_rois = self.pooling(psroipooled_cls_rois) # vote, ave the k*k psmap
        #---loc
        psroipooled_loc_rois = self.psroipooling_loc(rfcn_bbox, rois.view(-1, 5)) # shape (num_rois, 4*n_classes, k, k)
        ave_bbox_pred_rois = self.pooling(psroipooled_loc_rois) 
        ave_cls_score_rois = ave_cls_score_rois.squeeze() # shape (num_rois, 4nclasses )
        # ave_bbox_pred_rois = ave_bbox_pred_rois.squeeze()  
        cls_score_pred = F.softmax(ave_cls_score_rois, dim=1) #(n_rois, n_classes)

        if self.training and not self.class_agnostic:    
            # select the corresponding columns according to roi labels
            bbox_pred_view = ave_bbox_pred_rois.view(ave_bbox_pred_rois.size(0), int(ave_bbox_pred_rois.size(1) / 4), 4)
            # shape (n_rois, n_classes, 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            ave_bbox_pred_rois = bbox_pred_select.squeeze(1) # ( n_rois, 4 ) 

        if self.training:
            # classification loss
            self.RCNN_loss_cls = F.cross_entropy(ave_cls_score_rois, rois_label)
            # bounding box regression L1 loss
            self.RCNN_loss_bbox = _smooth_l1_loss(ave_bbox_pred_rois, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_score_pred.view(batch_size, rois.size(1), -1)
        bbox_pred = ave_bbox_pred_rois.view(batch_size, rois.size(1), -1)
        if self.training:
            rpn_loss_cls.unsqueeze_(0)
            rpn_loss_bbox.unsqueeze_(0)
            self.RCNN_loss_cls.unsqueeze_(0)
            self.RCNN_loss_bbox.unsqueeze_(0)
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, self.RCNN_loss_cls, self.RCNN_loss_bbox, rois_label

    # def train_rfcn(self, train=True, lr_decay=0.01):
    #     if train:
    #         self.RCNN_base

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RFCN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RFCN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RFCN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
        # print(self.state_dict().keys())
