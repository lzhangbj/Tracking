from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.resnet import resnet_rfcn, resnet101
from model.faster_rcnn.stmm_bases import STMM_Cell, STMM_RFCN


import random
import torchvision.models as models
import numpy as np
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from model.psroi_pooling.modules.psroi_pool import PSRoIPool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb





class STMM(nn.Module):
	def __init__(self, classes, class_agnostic, rfcn_pretrained=False, rfcn_pretrained_path=None):
		'''
			we assume you must use resnet101 pretrained model
			choose to use rfcn pretrained model's new_conv_1 module to init stmm or not
		'''
		super(STMM, self).__init__()
		self.classes = classes
		self.n_classes = len(classes)
		self.resnet101_pretrained_path = 'data/pretrained_model/resnet101_caffe.pth'
		self.rfcn_pretrained_path = None
		self.rfcn_pretrained = rfcn_pretrained
		if self.rfcn_pretrained:
			assert rfcn_pretrained_path is not None, "please input rfcn pretrained path"
			self.rfcn_pretrained_path = rfcn_pretrained_path
		self.dout_base_model = 1024
		self.class_agnostic = class_agnostic
		# modules bottom -> top
		self.Base = None
		self.RPN  = _RPN(self.dout_base_model)
		self.Proposal_target = _ProposalTargetLayer(self.n_classes)
		self.Top = None
		self.Cell = STMM_Cell()
		self.RFCN = STMM_RFCN(classes, class_agnostic) 

	def _init_modules(self):
		################################# set the resnet101 moudle ##############################
		resnet = resnet101()
		print("Loading resnet101 pretrained weights from %s" %(self.resnet101_pretrained_path))
		state_dict = torch.load(self.resnet101_pretrained_path)
		resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

		# Build resnet.
		self.Base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
			resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3) # output (batch, 1024 )
		self.Top = nn.Sequential(resnet.layer4)  # output (batch, 2048)
		# self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
		# if self.class_agnostic:
		#   self.RCNN_bbox_pred = nn.Linear(2048, 4)
		# else:
		#   self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)   set these in stmm_rfcn

		# Fix blocks zl pretrained
		for p in self.Base[0].parameters(): p.requires_grad=False
		for p in self.Base[1].parameters(): p.requires_grad=False

		assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
		if cfg.RESNET.FIXED_BLOCKS >= 3:
			for p in self.Base[6].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 2:
			for p in self.Base[5].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 1:
			for p in self.Base[4].parameters(): p.requires_grad=False
		def set_bn_fix(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				for p in m.parameters(): p.requires_grad=False
		self.Base.apply(set_bn_fix)
		self.Top.apply(set_bn_fix)         

	def forward(self, im_data, im_info, gt_boxes, num_boxes):
		'''
		@im_data: (batch_size, vid_len, 3, h, w)
		@im_info: (batch_size, 3) (h,w,scale)
		@gt_boxes:(batch_size, vid_len, max_num_boxes, 5)
		@num_boxes:(batch_size, vid_len, 1)
		'''
		assert im_info.size() == (1,3), im_info.size()
		assert im_data.size()[:3] == (1, 7, 3),  im_data.size()
		assert gt_boxes.size()[:2] == (1, 7), gt_boxes.size()
		batch_size = im_data.size(0)
		vid_len = im_data.size(1)
		C = im_data.size(2)
		H = im_data.size(3)
		W = im_data.size(4)
		Max_num_boxes = gt_boxes.size(2)

		im_info = im_info.data
		im_info = im_info.repeat(1,vid_len).view(-1,3)
		im_data = im_data.data.view(-1, C, H, W)
		gt_boxes = gt_boxes.data.view(-1, Max_num_boxes, 5)
		num_boxes = num_boxes.data.view(-1)

		# feed image data to base model to obtain base feature map
		base_feat = self.Base(im_data) # (batch*vid_len, 1024, h, w)
		# feed base feature map tp RPN to obtain rois
		rois, rpn_loss_cls, rpn_loss_bbox = self.RPN(base_feat, im_info, gt_boxes, num_boxes) 
		# rois of shape (batch*vid_len, max_num, 5) the 1st among 5 is batch*vid_len index\
		# losses are both numbers

		# if it is training phrase, then use ground truth bboxes for refining
		# sample boxes from both proposed and groundtruth boxes
		if self.training:
			roi_data = self.Proposal_target(rois, gt_boxes, num_boxes) 
			rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data # rois shape (batch*vid, max_num, 5)

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

		
		base_feat = self._head_to_tail(base_feat) # h and w halved (batch*vid, 2048, h, w)
		###### stmm ######
		base_feat = base_feat.view(batch_size, vid_len, base_feat.size(1), base_feat.size(2), base_feat.size(3)).transpose(1,0).contiguous()
		M = base_feat.new(vid_len, batch_size, 1024, base_feat.size(3), base_feat.size(4)).zero_()
		prev_F = base_feat.new(batch_size,2048, base_feat.size(3), base_feat.size(4)).zero_()
		prev_M = base_feat.new(batch_size,1024, base_feat.size(3), base_feat.size(4)).zero_()
		for i in range(vid_len):
			M[i] = self.Cell(base_feat[i], prev_F, prev_M, i)
			prev_F=base_feat[i]
			prev_M=M[i]

		rois = Variable(rois) 
		cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox = \
					self.RFCN(\
						M.view(-1, 1024, base_feat.size(3), base_feat.size(4)),\
						rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws)
		rois_label = rois_label.data.view(batch_size, vid_len, -1)
		cls_prob.view_(batch_size, vid_len, cls_prob.size(1), cls_prob.size(2))
		bbox_pred.view_(batch_size, vid_len, bbox_pred.size(1), bbox_pred.size(2))
		if self.training:
			rpn_loss_cls.unsqueeze_(0)
			rpn_loss_bbox.unsqueeze_(0)
			RCNN_loss_cls.unsqueeze_(0)
			RCNN_loss_bbox.unsqueeze_(0)

		return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

	def train(self, mode=True):
		# Override train so that the training mode is set as we want
		nn.Module.train(self, mode)
		if mode:
			# Set fixed blocks to be in eval mode
			self.Base.eval()
			self.Base[5].train()
			self.Base[6].train()  
			self.RFCN.train()

			def set_bn_eval(m):
				classname = m.__class__.__name__
				if classname.find('BatchNorm') != -1:
					m.eval()

			self.Base.apply(set_bn_eval) 
			self.Top.apply(set_bn_eval)

	def _head_to_tail(self, pool5):
		fc7 = self.Top(pool5)#.mean(3).mean(2)
		return fc7

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

		normal_init(self.RPN.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RPN.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
		normal_init(self.RPN.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
		############################### set the stmm cell module #################################
		if self.rfcn_pretrained:
			print("loading rfcn new_conv_1 weights from %s" %(self.rfcn_pretrained_path))
			state_dict = torch.load(self.rfcn_pretrained_path)
			for k,v in state_dict.items():
				if k=="conv_new_1/weight":
					w = v
				elif k=="conv_new_1/bias":
					b = v
			self.Cell.W_pretrained_init(w, b)

	def create_architecture(self):
		self._init_modules()
		self._init_weights()
