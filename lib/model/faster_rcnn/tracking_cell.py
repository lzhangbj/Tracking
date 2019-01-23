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
from model.faster_rcnn.resnet import resnet101
from model.nms.nms_wrapper import nms

import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from model.psroi_pooling.modules.psroi_pool import PSRoIPool
from model.faster_rcnn.rfcn import Conv2d
from model.faster_rcnn.tracking_utils import *
from model.rpn.bbox_transform import *

import torch.cuda as cutorch
import sys
import gc
from subprocess import call


def gpuStats(message=None):
	if message is None:
		message = ''
	sys.stdout.write("{}".format(message))
	for i in range(cutorch.device_count()):
		sys.stdout.write("\t{}: {:d}/{:d}\n".format(cutorch.get_device_name(i), int(cutorch.memory_allocated(i)*1e-6), int(cutorch.max_memory_allocated(i)*1e-6)))


DEVICE1 = torch.device('cuda:0')
DEVICE2 = torch.device('cuda:1')
DEVICE3 = torch.device('cuda:2')
DEVICE4 = torch.device('cuda:3')

class TrackingLSTMCell(nn.Module):
	"""
	Generate a convolutional GRU cell
	"""
	def __init__(self, in_channels=2048, out_channels=512, kernel_size=5, capacity=5):
		'''
		settings of W are the same as conv_new_1 in rfcn
		'''
		super(TrackingLSTMCell, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.capacity = capacity
		self.kernel_size = kernel_size
		padding = int(kernel_size-1)/2
		stride = 1
		self.k=2
		self.Wz = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=padding)
		self.Wr = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=padding)
		self.W = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=padding)        

		self.Uz = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=padding)
		self.Ur = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=padding)
		self.U = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=padding)

		self.content = torch.zeros((self.capacity,), dtype=torch.int32)

	def W_pretrained_init(self, weights=None, bias=None):
		self.Wz.weight.data.fill_(weights)
		self.Wz.bias.data.fill_(bias)        
		self.Wr.weight.data.fill_(weights)
		self.Wr.bias.data.fill_(bias)        
		self.W.weight.data.fill_(weights)
		self.W.bias.data.fill_(bias)

	def _BatchNormStar(self, input, K=3):
		'''
		@input: (batch, channel, h, w)
		return
		@input: (batch, channel, h, w) each value in range(0, 1)
		'''
		# assert input >= 0
		mu_X = torch.mean(input, dim=0).unsqueeze(0) # adding batch dimension for broadcasting in comparing with input
		std_X = torch.std(input, dim=0).unsqueeze(0)
		X_limit = mu_X + K*std_X
		temp = (input>X_limit)
		# temp = torch.nonzero(temp)
		# input[temp]=1
		input = torch.where(temp, input.new(1).fill_(1), input)
		del temp
		torch.cuda.empty_cache()
		return input

	def get_content(self):
		'''
			get the bool value of each content of the capacity
		''' 
		return self.content

	def forward(self, curr_F, prev_M, content):
		'''
		curr_F: current input
		( capacity, in_channel, H, W )
		prev_M: previous output 
		( capacity, out_channel, H, W )
		content
		one hot torch vector, same shape as self.content
		'''
		# get batch and spatial sizes
		assert curr_F.size()[:2] == (self.capacity, self.in_channels), curr_F.size()[:2]
		assert prev_M.size()[:2] == (self.capacity, self.out_channels), prev_M.size()[:2]

		self.content = content
		# aligned_prev_M = prev_M
		zt = self._BatchNormStar(\
			F.relu(\
				self.Wz(curr_F) + self.Uz(prev_M)
				)
			)
		rt = self._BatchNormStar(\
			F.relu(\
				self.Wr(curr_F) + self.Ur(prev_M)
				)
			)
		M = F.relu(\
				self.W(curr_F) + self.U(prev_M*rt)
				)
		curr_M = (1-zt)*prev_M + zt*M
		assert curr_M.size(1) == self.out_channels, curr_M.size()

		return curr_M


class TrackingCell(nn.Module):
	""" faster RFCN """
	def __init__(self, classes, class_agnostic, pretrained_rfcn=None, capacity=5):
		super(TrackingCell, self).__init__()
		
		self.classes = classes
		self.n_classes = len(classes)
		self.class_agnostic = class_agnostic
		self.pretrained_rfcn = pretrained_rfcn
		#################################### set up RFCN part ############################
		# loss
		self.RFCN_loss_cls = 0
		self.RFCN_loss_bbox = 0
		# define base model
		self.resnet101_pretrained_path = 'data/pretrained_model/resnet101_caffe.pth'
		self.RFCN_base = None
		# self.RFCN_top  = None
		self.dout_base_model = 512
		# define rpn
		self.RFCN_rpn = _RPN(self.dout_base_model)
		self.RFCN_proposal_target = _ProposalTargetLayer(self.n_classes)
		# define rfcn
		self.k = 7
		# self.conv_new_1 = Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, relu=True)
		self.RFCN_cls_net = nn.Conv2d(512,self.n_classes*7*7, [1,1], padding=0, stride=1)
		self.psroipooling_cls = PSRoIPool(pooled_height=self.k, pooled_width=self.k,
										spatial_scale= 1/16.0, group_size=self.k, 
										output_dim=self.n_classes)
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
		self.pooling = nn.AvgPool2d(kernel_size=self.k, stride=self.k)

		##################################### set up postprocessing parameters ####################
		# cls threshold, we only consider boxes having cls score > thresh as final result
		# self.cls_thresh = 0.7
		# nms threshold, for boxes having overlap > 0.7, we choose ones with higher cls score
		self.nms_thresh = 0.7

		##################################### set up tracking part ###########################
		# set up lstm part
		self.capacity = capacity
		self.dynamic_info_kernel = 5
		# here the 3 is x move, y move, correlation 
		self.tracking_lstm_cell = TrackingLSTMCell(in_channels=512+self.dynamic_info_kernel*self.dynamic_info_kernel*3, \
																	out_channels=512+ 2*(self.n_classes) , \
																	kernel_size=3, capacity=self.capacity)
		self.lstm_feature_k = 16
		# self.tracking_loc_lstm_cell = TrackingLSTMCell(in_channels=1024+2, out_channels=16, \
		#                                                           kernel_size=7, capacity=self.tracking_capacity)
		# set up tracking comparison network
		# here we set in_channels to 2*ori_channel, then need to apply this netowrk for capacity times.
		# this is to avoid re-training when we change capacity
		# self.tracking_loc_conv = nn.Conv2d(in_channels= (capacity+1) *self.n_classes *4, out_channels= 2*capacity,
		#                       kernel_size=3, stride=1, padding=1, bias=False)
		# this pooling will divide the tracking loc map into 2*2, pre region for each corner move
		self.tracking_loc_movement_pooling = nn.AvgPool2d(kernel_size=self.lstm_feature_k/2, stride=self.lstm_feature_k/2)
		# before tracking compare, we need to get the roi pooling/align of the roi feature
		self.tracking_cls_roi_align = _RoIPooling(self.lstm_feature_k, self.lstm_feature_k, 1.0/16.0) # 32 or 16
		# tracking cls compare conv out channel is 2, same or not, then we do a soft max and then average pooling
		# self.tracking_cls_compare_conv = Conv2d(in_channels=512*2, out_channels=1,kernel_size=3, same_padding=True)
		self.tracking_cls_compare_conv = nn.Conv2d(512*2, 1, 3, padding = 1, stride=1)          ################################### problem may be here, consider using psroi pool and features

		self.tracking_cls_compare_pooling = nn.AvgPool2d(kernel_size=self.lstm_feature_k, stride=self.lstm_feature_k)
		# self.tracking_cls_conv = nn.Conv2d(in_channels = (capacity+1) *self.n_classes, out_channels= 2*capacity,
		#                       kernel_size=3, stride=1, padding=1, bias=False)
		# tracking reg threshold, we track boxes only if their reg score > thresh
		self.tracking_loc_overlap_thresh = 0.2
		# tracking cls threshold, we tracki boxes only if their cls score > thresh
		self.tracking_cls_thresh = 0.8

		self.new_tracking_cls_thresh = 0.9

		self.tracking_cls_loss = None
		self.tracking_loc_loss = None
		self.tracking_predict_loc_loss = None

	def _init_modules(self):
		resnet = resnet101()

		# if self.pretrained == True:
		print("Loading pretrained weights from %s" %(self.resnet101_pretrained_path))
		state_dict = torch.load(self.resnet101_pretrained_path)
		resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

		# Build resnet.
		self.RFCN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
			resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3, resnet.layer4) # output (batch, 1024 )

		# Fix blocks zl pretrained
		for p in self.RFCN_base[0].parameters(): p.requires_grad=False
		for p in self.RFCN_base[1].parameters(): p.requires_grad=False

		assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
		if cfg.RESNET.FIXED_BLOCKS >= 3:
			for p in self.RFCN_base[6].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 2:
			for p in self.RFCN_base[5].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 1:
			for p in self.RFCN_base[4].parameters(): p.requires_grad=False

		

		def set_bn_fix(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				for p in m.parameters(): p.requires_grad=False

		self.RFCN_base.apply(set_bn_fix)
		self.RFCN_net = nn.Conv2d(2048, 512, kernel_size=3, padding=6, stride=1, dilation=6)
		self.RFCN_base.add_module("RFCN_net", self.RFCN_net)
		self.RFCN_base.add_module("resnet", resnet.relu)
		for p in self.RFCN_base[4].parameters(): p.requires_grad=False
		for p in self.RFCN_base[5].parameters(): p.requires_grad=False
		for p in self.RFCN_base[6].parameters(): p.requires_grad=False
		for p in self.RFCN_base[7].parameters(): p.requires_grad=False
		for p in self.RFCN_base[8].parameters(): p.requires_grad=False
		for p in self.RFCN_rpn.parameters(): p.requires_grad=False
		for p in self.RFCN_bbox_net.parameters(): p.requires_grad=False
		for p in self.RFCN_cls_net.parameters(): p.requires_grad=False
		# self.train_rfcn(train=False)

	def train(self, mode=True):
		# Override train so that the training mode is set as we want
		nn.Module.train(self, mode)
		if mode:
			# Set fixed blocks to be in eval mode
			self.RFCN_base.eval()
			# self.RFCN_bbox_net.eval()
			# self.RFCN_cls_net.eval()
			# self.RFCN_rpn.eval()
			# self.RFCN_base[5].train()
			# self.RFCN_base[6].train()   # zl why
			# self.RFCN_base[7].train()
			# self.RFCN_base[8].train()
			# self.RFCN_bbox_net.train()
			# self.RFCN_cls_net.train()
			self.tracking_lstm_cell.train()
			self.tracking_cls_compare_conv.train()


			def set_bn_eval(m):
				classname = m.__class__.__name__
				if classname.find('BatchNorm') != -1:
					m.eval()

			self.RFCN_base.apply(set_bn_eval) 

	def train_rfcn(self, train=True, lr = 0.001, lr_decay=0.1):
		def make_train(layer, train=True):
			for p in layer.parameters():
				p.requires_grad = train
		make_train(self.RFCN_base[6])
		make_train(self.RFCN_base[5])
		make_train(self.RFCN_base[4])
		make_train(self.RFCN_cls_net)
		make_train(self.RFCN_bbox_net)
		assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
		if cfg.RESNET.FIXED_BLOCKS >= 3:
			make_train(self.RFCN_base[6], False)
		if cfg.RESNET.FIXED_BLOCKS >= 2:
			make_train(self.RFCN_base[5], False)
		if cfg.RESNET.FIXED_BLOCKS >= 1:
			make_train(self.RFCN_base[4], False)
		return None

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

		normal_init(self.tracking_lstm_cell.Wz, 0, 0.01)
		normal_init(self.tracking_lstm_cell.Wr, 0, 0.01)
		normal_init(self.tracking_lstm_cell.W, 0, 0.01)
		normal_init(self.tracking_lstm_cell.Uz, 0, 0.01)
		normal_init(self.tracking_lstm_cell.Ur, 0, 0.01)
		normal_init(self.tracking_lstm_cell.U, 0, 0.01)
		normal_init(self.tracking_cls_compare_conv, 0, 0.01)

		if self.pretrained_rfcn is not None:
			print("Loading pretrained weights from %s" %(self.pretrained_rfcn))
			state_dict = torch.load(self.pretrained_rfcn)
			model_dict = self.state_dict()
			dictt = {k:v for k,v in state_dict["model"].items() if k in self.state_dict()}
			assert len(dictt) > 0, len(dictt)
			model_dict.update(dictt)
			self.load_state_dict(model_dict)
			# print(self.state_dict().keys())
			# print(state_dict['model'].keys())
			# print(self.state_dict().keys())
			# for i in state_dict["model"].keys():
			#   if i in self.state_dict().keys():
			#       assert not torch.sum(self.state_dict()[i] != state_dict["model"][i].cpu())
				# if "conv_new_1" in i and "weight" in i:
					# print(self.state_dict()[i])

	def create_architecture(self):
		self._init_modules()
		self._init_weights()

	def forward(self, im_data, im_info, GT_boxes, num_boxes, input=None):
		'''
		@im_data: (batch_size, vid_len, 3, h, w)
		@im_info: (batch_size, 3) (h,w,scale)
		@gt_boxes:(batch_size, vid_len, max_num_boxes, 6)  x1, y1, x2 y2, trackid, cls
		@gt_boxes_track_ids: (batch_size, vid_len, max_num_boxes) track id of gt_boxes   X
		@num_boxes:(batch_size, vid_len)

		for memory saving reasons, we suggest batch__size to be 1
		'''
		assert im_info.size() == (1,3), im_info.size()
		assert im_data.size(0) == 1,  im_data.size()
		assert GT_boxes.size(0) == 1, GT_boxes.size()
		batch_size = im_data.size(0)
		vid_len = im_data.size(1)
		C = im_data.size(2)
		H = im_data.size(3)
		W = im_data.size(4)
		Max_num_boxes = GT_boxes.size(2)
		assert batch_size == 1, "batch_size can only be 1"
		self.tracking_cls_loss = torch.zeros(vid_len).cuda()
		self.tracking_loc_loss = torch.zeros(vid_len).cuda()
		self.tracking_predict_loc_loss = torch.zeros(vid_len).cuda()

		###########  combine batch size and vid_length shape into batch_size*vid_lenght shape  #################
		im_info = im_info.data
		im_info = im_info.repeat(1,vid_len).view(batch_size*vid_len,3)
		im_data = im_data.data.view(batch_size*vid_len, C, H, W)
		gt_boxes = torch.cat([GT_boxes.data[:, :, :, :4].view(batch_size*vid_len, Max_num_boxes, 4), GT_boxes.data[:, :, :, 5].view(batch_size*vid_len, Max_num_boxes, 1)], dim=2)
		num_boxes = num_boxes.data.view(batch_size*vid_len)
		gt_boxes_track_ids = GT_boxes.data[:, :, :, 4].view(batch_size*vid_len, Max_num_boxes).long()

		################################################################################################
		################################################################################################
		###################################### rfcn on all rois ########################################
		################################################################################################
		################################################################################################
		# feed image data to base model to obtain base feature map
		# rois of shape (vid_len, post_nms_top_N, 5) 
		# 5 = (vid_ind, x1, y1, x2, y2)
		base_feat = self.RFCN_base(im_data) # (batch, 1024, h, w)
		rois, rpn_loss_cls, rpn_loss_bbox = self.RFCN_rpn(base_feat, im_info, gt_boxes, num_boxes)
		rois_per_image = rois.size(1)
		# if it is training phrase, then use ground truth bboxes for refining
		# sample boxes from both proposed and groundtruth boxes
		if self.training:
			roi_data = self.RFCN_proposal_target(rois, gt_boxes, num_boxes)
			# rois_label    : (batch_size, rois_per_image)
			# rois          : (batch_size, rois_per_image, 5)  5 = (batch_id, x1, y1, x2, y2)
			# bbox_target   : b x rois_per_image x 4 blob of regression targets
			# rois_inside_ws: b x rois_per_image x 4 blob of loss weights
			# rois_outside_ws: 
			#       the weight will only apply on those box whose overlap with gtbox is high enough, which we regard as fg box
			# remind here we insert gt boxes into roi_data to ensure catching the gt boxes
			rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
			rois_per_image = rois.size(1)
			rois_label = Variable(rois_label.view(-1).long())
			rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
			rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
			rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
		else:
			# rois = rois.cuda(0)
			rois_label = None
			rois_target = None
			rois_inside_ws = None
			rois_outside_ws = None
			rpn_loss_cls = 0
			rpn_loss_bbox = 0
		rois = Variable(rois) # remove batch become shape  (vid_len, rois_per_image, 5)
		# base_feat = self.RFCN_top(base_feat) # h and w halved (batch, 2048, h, w)
		# do roi pooling based on predicted rois
		rfcn_cls = self.RFCN_cls_net(base_feat) # output (batch, k*k*n_classes, h, w)
		rfcn_bbox = self.RFCN_bbox_net(base_feat)  # output (batch, 8*k*k, h, w) 
		#----cls
		psroipooled_cls_rois = self.psroipooling_cls(rfcn_cls, rois.view(batch_size*vid_len*rois_per_image, 5)) # shape (num_rois,  n_classes, k, k)
		ave_cls_score_rois = self.pooling(psroipooled_cls_rois) # vote, ave the k*k psmap
		#---loc
		psroipooled_loc_rois = self.psroipooling_loc(rfcn_bbox, rois.view(batch_size*vid_len*rois_per_image, 5)) # shape (num_rois, 4*n_classes, k, k)
		ave_bbox_pred_rois = self.pooling(psroipooled_loc_rois)
		# post process the scores
		ave_cls_score_rois.squeeze_() # shape (num_rois, n_classes )
		cls_score_pred = F.softmax(ave_cls_score_rois, dim=1) #(n_rois, n_classes)

		# remind here we will only choose class_agnostic mode
		if self.training and not self.class_agnostic:    
			# select the corresponding columns according to roi labels
			bbox_pred_view = ave_bbox_pred_rois.view(ave_bbox_pred_rois.size(0), int(ave_bbox_pred_rois.size(1) / 4), 4)
			# shape (n_rois, n_classes, 4)
			bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
			ave_bbox_pred_rois = bbox_pred_select.squeeze(1) # ( n_rois, 4 ) 
		if self.training:
			# classification loss
			self.RFCN_loss_cls = F.cross_entropy(ave_cls_score_rois, rois_label)
			# bounding box regression L1 loss
			self.RFCN_loss_bbox = _smooth_l1_loss(ave_bbox_pred_rois, rois_target, rois_inside_ws, rois_outside_ws)

		cls_prob = cls_score_pred.view(batch_size, vid_len, rois_per_image, self.n_classes)      # (b, v, rois, n_classes)
		bbox_pred = ave_bbox_pred_rois.view(batch_size, vid_len, rois_per_image, 4) # (b, v, rois, 4)
		###############################################################################################
		###############################################################################################
		################################  COME ON ! LET'S FUCKING TRACK ###############################
		###############################################################################################
		###############################################################################################
		if input is not None:
			predict_track_M,prev_track_features,prev_track_ids,prev_track_cls,prev_track_roi = input
		else:
			predict_track_M = psroipooled_cls_rois.new(self.capacity, 512+2*(self.n_classes),self.lstm_feature_k, self.lstm_feature_k).zero_()
			prev_track_features = psroipooled_cls_rois.new(self.capacity, 512, self.lstm_feature_k, self.lstm_feature_k)
			prev_track_ids = -gt_boxes_track_ids.new(self.capacity).fill_(1)
			prev_track_cls = gt_boxes_track_ids.new(self.capacity).zero_()
			prev_track_roi = rois.new(self.capacity, 4).zero_()

		# cuda long -> prev_track_ids
		# cuda float -> prev_track_M


		video_track_rois = predict_track_M.new(vid_len, self.capacity, 4)
		video_track_ids = prev_track_ids.new(vid_len, self.capacity)
		video_track_cls = predict_track_M.new(vid_len, self.capacity)
		video_track_old_num = prev_track_ids.new(vid_len)
		video_track_new_num = prev_track_ids.new(vid_len)
		video_track_predict_rois = predict_track_M.new(vid_len, self.capacity, 4)
		video_track_predict_rois_id = prev_track_ids.new(vid_len, self.capacity)

		for i in range(vid_len):
			# var name with obj: the non-1 obj among the capacity objs
			track_old_num = 0
			track_new_num = 0
			track_lost_num = 0
			prev_track_obj_inds = (prev_track_ids!=-1).nonzero().view(-1)
			track_obj_num = prev_track_obj_inds.size(0)

			if track_obj_num != 0:
				# prev_track_obj_ids = prev_track_ids[prev_track_obj_inds]
				predict_track_obj_M =  predict_track_M[prev_track_obj_inds]
				prev_track_obj_cls = prev_track_cls[prev_track_obj_inds]
				prev_track_obj_roi = prev_track_roi[prev_track_obj_inds]
				############################################################################################################
				############################ first get predict tracking obj's roi loc  #####################################
				############################    and calculate prediction location loss #####################################
				############################################################################################################
				predict_track_obj_movement_map = predict_track_M.new(track_obj_num, 2,self.lstm_feature_k, self.lstm_feature_k)
				for j in torch.arange(track_obj_num).cuda():
					predict_track_obj_movement_map[j] = predict_track_obj_M[j, 512+prev_track_obj_cls[j]*2:512+prev_track_obj_cls[j]*2+2]
				predict_track_obj_Movement = self.tracking_loc_movement_pooling(predict_track_obj_movement_map)
				predict_track_obj_dx1 = torch.mean(predict_track_obj_Movement[:, 0, 0, :], dim=1, keepdim=True)
				predict_track_obj_dy1 = torch.mean(predict_track_obj_Movement[:, 1, :, 0], dim=1, keepdim=True)
				predict_track_obj_dx2 = torch.mean(predict_track_obj_Movement[:, 0, 1, :], dim=1, keepdim=True)
				predict_track_obj_dy2 = torch.mean(predict_track_obj_Movement[:, 1, :, 1], dim=1, keepdim=True)
				predict_track_obj_rela_move = torch.cat([predict_track_obj_dx1, predict_track_obj_dy1, predict_track_obj_dx2, predict_track_obj_dy2], dim=1)
				predict_track_obj_roi = prev_track_obj_roi+predict_track_obj_rela_move
				predict_track_obj_roi = clip_tracking_boxes(predict_track_obj_roi, im_info[i])
				# check the validation of track rois
				predict_track_obj_roi, predict_track_obj_roi_valid_inds = tracking_boxes_validation_check(predict_track_obj_roi)
				predict_track_old_roi = predict_track_obj_roi[predict_track_obj_roi_valid_inds, :]
				video_track_predict_rois[i, :track_obj_num] = predict_track_obj_roi
				video_track_predict_rois_id[i, :track_obj_num] = prev_track_ids[prev_track_obj_inds]
				# in testing mode, we get old indices now
				if not self.training:
					prev_track_old_inds = prev_track_obj_inds[predict_track_obj_roi_valid_inds]
					track_old_num = prev_track_old_inds.size(0)
					prev_track_lost_inds = [t for t in range(prev_track_obj_inds.size(0)) if t not in predict_track_obj_roi_valid_inds]
					prev_track_lost_inds = prev_track_obj_inds[prev_track_lost_inds]
					track_lost_num = prev_track_lost_inds.size(0)
				# and calculate prediction loss
				# remind here we set a hard loss for the network to find the predict bbox
				# but there is a chance that the predict box is VERY hard to get
				# in which case we should consider to put this loss after the cls compare network
				# after which we only look for box with compare score >= threshold
				# 
				if self.training:
					# find prev_boxes's ind in current gt_boxes
					curr_track_obj_gt_rois = predict_track_M.new(predict_track_obj_roi.size()).zero_()
					curr_track_obj_gt_rois_mask = predict_track_M.new(track_obj_num).zero_()
					for j in torch.arange(track_obj_num).cuda():
						t = (gt_boxes_track_ids[i, :num_boxes[i]]==prev_track_ids[prev_track_obj_inds[j]]).nonzero()
						# if prev box still exists
						if t.numel() != 0:
							assert t.numel()==1, [ t, gt_boxes_track_ids[i, :num_boxes[i]] ]
							t = t.squeeze()
							curr_track_obj_gt_rois_mask[j] = 1
							curr_track_obj_gt_rois[j, :] = gt_boxes[i, t, :4]
						else:
							predict_track_obj_roi[j,:] = 0
					# transfer both boxes into log value
					gt_obj_target = bbox_transform_batch(curr_track_obj_gt_rois, curr_track_obj_gt_rois)
					predict_target =  bbox_transform_batch(predict_track_obj_roi, curr_track_obj_gt_rois)
					self.tracking_predict_loc_loss[i] = 10000*_smooth_l1_loss(predict_target, gt_obj_target)
					# replace predict tracking roi with gt rois
					# and further drop the nonexist rois in prev_track_obj_inds
					curr_track_old_gt_rois_inds = curr_track_obj_gt_rois_mask.nonzero().view(-1)
					predict_track_old_roi = curr_track_obj_gt_rois[curr_track_old_gt_rois_inds]
					prev_track_old_inds = prev_track_obj_inds[curr_track_old_gt_rois_inds]
					gt_old_target = gt_obj_target[curr_track_old_gt_rois_inds]
					track_old_num = prev_track_old_inds.size(0)
					prev_track_lost_inds = prev_track_obj_inds[(curr_track_obj_gt_rois_mask==0).nonzero().view(-1)]
					track_lost_num = prev_track_lost_inds.size(0)
					# curr_track_old_rois = predict_track_old_roi
				###################################################################################################################
				########################### second find tracking obj's specific roi                       #########################
				########################### by finding highest cls score map in predict roi's nearby rois #########################
				###################################################################################################################
				# first get our compare output cls feature from lstm
				if track_old_num != 0:
					predict_track_old_M_feature = predict_track_M[prev_track_old_inds, :512, :, :]
					# then find our overlap features
					curr_track_old_rois = predict_track_M.new(track_old_num, 4)
					curr_track_old_roi_inds = prev_track_ids.new(track_old_num).zero_()

					rois_overlaps = bbox_overlaps(rois[i, :, 1:], predict_track_old_roi)
					rois_overlaps_max, rois_max_predict_inds  = rois_overlaps.max(dim=1)
					# prepare something for training loss
					if self.training:
						compare_results = predict_track_M.new(track_old_num, 20).zero_()
						compare_gts = prev_track_ids.new(track_old_num).zero_()
						gt_old_rois_inds = prev_track_ids.new(track_old_num).zero_()
					for j in torch.arange(track_old_num).cuda():
						# for each tracking old obj, find top 20 boxes to be candidate
						rois_j_scores, rois_j_inds =  torch.topk(rois_overlaps[:, j], 20)
						rois_j_inds = rois_j_inds.long().view(-1)
						if self.training:
							compare_gt_ind = (rois_j_scores == 1).nonzero().view(-1).long()
							assert compare_gt_ind.numel()>=1, compare_gt_ind
							if compare_gt_ind.numel() > 1: 													# here it is strange
								print("strangely lucky {}".format(compare_gt_ind.numel()))
								compare_gt_ind = compare_gt_ind[:1]
							compare_gts[j] = compare_gt_ind
							gt_old_rois_inds[j] = rois_j_inds[compare_gt_ind]

						roi_predict_candidates_j_append_rois = rois[i, rois_j_inds]
						roi_predict_candidates_j_append_rois[:, 0] = i
						roi_predict_candidates_j_pooled_feat = self.tracking_cls_roi_align(base_feat, roi_predict_candidates_j_append_rois)

						assert roi_predict_candidates_j_pooled_feat.size() == (20, 512, self.lstm_feature_k, self.lstm_feature_k),\
							roi_predict_candidates_j_pooled_feat.size()
						compare_feature = predict_track_old_M_feature[j].unsqueeze(0).expand(20, 512, self.lstm_feature_k, self.lstm_feature_k)
						compare_input = torch.cat([compare_feature, roi_predict_candidates_j_pooled_feat], dim=1)
						# print(roi_predict_candidates_j_pooled_feat)
						# print(compare_input)
						# print(torch.sum(compare_input!=compare_input))
						# compare_input.zero_()
						compare_result = self.tracking_cls_compare_conv(compare_input)
						print(roi_predict_candidates_j_pooled_feat)
						# print(compare_result)
						compare_result = self.tracking_cls_compare_pooling(compare_result).view(-1)
						# print(self.tracking_cls_compare_conv.weight.data)
						if self.training:
							compare_results[j] = compare_result
						tracking_roi_predict_box_ind_j     = 19 - torch.argmax(compare_result[torch.arange(19, -1, -1)])
						curr_track_old_rois[j] = roi_predict_candidates_j_append_rois[tracking_roi_predict_box_ind_j, 1:]
						curr_track_old_roi_inds[j]   = rois_j_inds[tracking_roi_predict_box_ind_j]
					# then we calculate loss
					if self.training:
						# print(compare_results)
						# print(compare_gts)
						# print(compare_results)
						# print(compare_gts)
						self.tracking_cls_loss[i] = F.cross_entropy(compare_results, compare_gts)
						print(self.tracking_cls_loss[i])
						tracking_predict_target =  bbox_transform_batch(curr_track_old_rois, predict_track_old_roi)
						self.tracking_loc_loss[i] = _smooth_l1_loss(tracking_predict_target, gt_old_target)
						curr_track_old_roi_inds = gt_old_rois_inds
						curr_track_old_rois = predict_track_old_roi
					# get the rois left for rfcn to predict new tracking boxes
					# keep is the mask leading to left rois indices
					rois_tracking_based = predict_track_M.new(rois.size(1), 5).zero_()
					rois_tracking_based[:, :4] = rois[i,:, 1:]
					rois_tracking_based[curr_track_old_roi_inds, 4] = 1  ########################### here, when necesary, try to sort the rois, high->low
					keep = nms(rois_tracking_based, self.nms_thresh)
					keep = keep.view(-1).long()
					keep = torch.tensor([t for t in keep if t not in curr_track_old_roi_inds]).long().cuda()
				else:
					keep = torch.arange(rois_per_image).cuda()
			else:
				old_boxes_num = 0
				lost_boxes_num = 0
				keep = torch.arange(rois_per_image).cuda()
			###################################################################################################################
			################## step 3: choose the new bbox in all left rois ###################################################
			##################         and calculate loss  yes !            ###################################################
			###################################################################################################################
			###############################
			### this is to choose new boxes
			###############################
			if not self.training:
				cls_prob_keep = cls_prob[0, i, keep, :]
				bbox_pred_keep = bbox_pred[0, i, keep, :]
				rois_keep   = rois[i, keep]
				cls_prob_keep, cls_prob_keep_class = torch.max(cls_prob_keep, dim=1)
				rois_keep = torch.cat([rois_keep, cls_prob_keep.unsqueeze(1)], dim=1)
				rois_aft_2_nms_inds = nms(rois_keep, self.nms_thresh).view(-1).long()

				cls_prob_aft_nms = cls_prob_keep[rois_aft_2_nms_inds]
				cls_prob_aft_nms_cls = cls_prob_keep_class[rois_aft_2_nms_inds]
				new_rois_in_aft_nms_inds = (cls_prob_aft_nms>=self.new_tracking_cls_thresh).nonzero().view(-1)
				_, new_rois_in_aft_nms_inds = torch.topk(cls_prob_aft_nms[new_rois_in_aft_nms_inds])  # sorted
				predict_new_boxes_cls = cls_prob_aft_nms_cls[new_rois_in_aft_nms_inds]
				new_rois_in_keep_inds = rois_aft_2_nms_inds[new_rois_in_aft_nms_inds]
				track_new_num = new_rois_in_keep_inds.size(0)

				new_rois_bbox_pred = bbox_pred_keep[new_rois_in_keep_inds, :]
				new_rois_bbox = rois[i, keep[new_rois_in_keep_inds], 1:5]
				new_rois_inds = keep[new_rois_in_keep_inds]

				# calc the loss
				# restrict boxes num below maximum
				assert self.capacity >= track_old_num, [self.capacity , track_old_num]
				if track_new_num > self.capacity -  track_old_num:
					track_new_num = self.capacity-track_old_num
					if track_new_num != 0:
						new_rois_bbox = new_rois_bbox[:track_new_num, :]
						new_rois_bbox_pred = new_rois_bbox_pred[:track_new_num]
						# new_rois_inds = 
				if track_new_num != 0:
					new_rois_bbox = new_rois_bbox.unsqueeze(0)
					if cfg.TEST.BBOX_REG:
						# Apply bounding-box regression deltas
						if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
							new_rois_bbox_pred = new_rois_bbox_pred*torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
									   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						predict_new_boxes = bbox_transform_inv(new_rois_bbox, new_rois_bbox_pred.unsqueeze(0), 1)
						predict_new_boxes = clip_boxes(predict_new_boxes, im_info, 1) # (batch, num_rois, 4)
					predict_new_boxes = predict_new_boxes.squeeze()
					if track_old_num != 0:
						track_ids_choices = torch.tensor([t for t in torch.arange(self.capacity).cuda() if t not in prev_track_ids[prev_track_old_inds]]).cuda()
					else:
						track_ids_choices = torch.arange(self.capacity).cuda()
					# track_ids_choices, _ = torch.sort(track_ids_choices)
					new_boxes_track_ids = track_ids_choices[:track_new_num]
				# finally we get the new predicted bboxes' roi
				# with shape ( len(new_rois_inds), 4 )
			if self.training:
				# remind ":num_boxes[i]" we assume the true gt boxes are all in the front colums of gt_boxes
				predict_new_boxes_inds = torch.tensor([t for t in torch.arange(num_boxes[i]).cuda() if gt_boxes_track_ids[i, t] not in prev_track_ids]).long().cuda()
				track_new_num = predict_new_boxes_inds.numel()
				if track_new_num > self.capacity-track_old_num:
					track_new_num = self.capacity-track_old_num
					predict_new_boxes_inds = predict_new_boxes_inds[:track_new_num]
				predict_new_boxes = gt_boxes[i, predict_new_boxes_inds, :4]
				predict_new_boxes_cls = gt_boxes[i, predict_new_boxes_inds, 4]
				new_boxes_track_ids = gt_boxes_track_ids[i, predict_new_boxes_inds]
			# get the new boxes features
			if track_new_num != 0:
				predict_new_boxes_append = predict_track_M.new(track_new_num, 5).zero_()
				predict_new_boxes_append[:, 0] = i
				predict_new_boxes_append[:, 1:] = predict_new_boxes
				curr_track_new_features = self.tracking_cls_roi_align(base_feat, predict_new_boxes_append)
			###############################     
			### this is to choose old boxes
			###############################
			if track_old_num != 0:
				if not self.training:
					# calculate the trakcing boxes rois 
					old_rois_bbox_pred = bbox_pred[0, i, curr_track_old_roi_inds, :]
					old_rois_bbox = curr_track_old_rois.unsqueeze(0)
					if len(old_rois_bbox_pred.size())==2:
						old_rois_bbox_pred = old_rois_bbox_pred.unsqueeze(0)
					if cfg.TEST.BBOX_REG:
						# Apply bounding-box regression deltas
						if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
							old_rois_bbox_pred = old_rois_bbox_pred*torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
									   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						predict_old_boxes = bbox_transform_inv(old_rois_bbox, old_rois_bbox_pred, 1)
						predict_old_boxes = clip_boxes(predict_old_boxes, im_info, 1) # (batch, num_rois, 4)
					curr_track_old_rois = predict_old_boxes.squeeze(0)
				predict_old_boxes_cls = prev_track_cls[prev_track_old_inds]
				# get the new boxes features
				predict_old_boxes_append = predict_track_M.new(track_old_num, 5).zero_()
				predict_old_boxes_append[:, 0] = i
				predict_old_boxes_append[:, 1:] = curr_track_old_rois
				curr_track_old_features = self.tracking_cls_roi_align(base_feat, predict_old_boxes_append)
			####################
			# get track features
			if track_old_num != 0 and track_new_num != 0:
				curr_track_obj_features = torch.cat([curr_track_old_features, curr_track_new_features], dim=0)
			elif track_new_num != 0:
				curr_track_obj_features = curr_track_new_features
			elif track_old_num != 0:
				curr_track_obj_features = curr_track_old_features
			else:
				curr_track_obj_features = torch.tensor([]).cuda()
			curr_track_features = predict_track_M.new(self.capacity, 512, self.lstm_feature_k, self.lstm_feature_k).zero_()
			curr_track_features[:(track_old_num+track_new_num)] = curr_track_obj_features
			########################
			# get dynamic infomation
			track_new_boxes_dynamic_infos = predict_track_M.new(self.capacity-track_old_num, \
											3*self.dynamic_info_kernel*self.dynamic_info_kernel, self.lstm_feature_k, self.lstm_feature_k).zero_()
			if track_old_num != 0:
				track_old_boxes_dynamic_info = TrackingDynamicInfos(prev_track_features[prev_track_old_inds, :, :, :].cpu(), \
																prev_track_roi[prev_track_old_inds, :].cpu(), \
																curr_track_old_features.cpu(), \
																curr_track_old_rois.cpu(), \
																self.dynamic_info_kernel).cuda()
				track_boxes_dynamic_infos = torch.cat([track_old_boxes_dynamic_info,track_new_boxes_dynamic_infos], dim=0)
			else:
				track_boxes_dynamic_infos = track_new_boxes_dynamic_infos

			#####################
			# lstm input now
			curr_lstm_input = torch.cat([curr_track_features, track_boxes_dynamic_infos], dim=1)  
			if track_old_num != 0:
				predict_track_M[:track_old_num] = predict_track_M[prev_track_old_inds]
				prev_track_ids[:track_old_num] = prev_track_ids[prev_track_old_inds]
				prev_track_cls[:track_old_num] = prev_track_cls[prev_track_old_inds]
				prev_track_roi[:track_old_num] = curr_track_old_rois
			predict_track_M[track_old_num:] = 0
			if track_new_num != 0:
				prev_track_ids[track_old_num:(track_old_num+track_new_num)] = new_boxes_track_ids
				prev_track_cls[track_old_num:(track_old_num+track_new_num)] = predict_new_boxes_cls
				prev_track_roi[track_old_num:(track_old_num+track_new_num)] = predict_new_boxes
			prev_track_ids[(track_old_num+track_new_num):] = -1
			prev_track_cls[(track_old_num+track_new_num):] = 0
			prev_track_roi[(track_old_num+track_new_num):] = 0
			prev_track_features= curr_track_features

			################################
			# lstm
			predict_track_M = self.tracking_lstm_cell(curr_lstm_input, predict_track_M, self.capacity)

			video_track_rois[i] = prev_track_roi
			video_track_ids[i] = prev_track_ids
			video_track_cls[i] = prev_track_cls
			video_track_old_num[i] = track_old_num
			video_track_new_num[i] = track_new_num
		if self.training:
			output = [predict_track_M,\
						prev_track_features,\
						prev_track_ids,\
						prev_track_cls,\
						prev_track_roi]
		else:
			output = (video_track_rois, video_track_ids, video_track_cls, \
						video_track_old_num, video_track_new_num, video_track_predict_rois, video_track_predict_rois_id)
		EPISILON = 1e-6
		self.tracking_cls_loss_num = self.tracking_cls_loss.nonzero().size(0)
		self.tracking_cls_loss = self.tracking_cls_loss.sum() / (self.tracking_cls_loss_num + EPISILON) 
		self.tracking_cls_loss = self.tracking_cls_loss.view(1)

		self.tracking_predict_loc_loss_num = self.tracking_predict_loc_loss.nonzero().size(0)
		self.tracking_predict_loc_loss = self.tracking_predict_loc_loss.sum() / (self.tracking_predict_loc_loss_num + EPISILON) 
		self.tracking_predict_loc_loss = self.tracking_predict_loc_loss.view(1)

		self.tracking_loc_loss_num = self.tracking_loc_loss.nonzero().size(0)
		self.tracking_loc_loss = self.tracking_loc_loss.sum() / (self.tracking_loc_loss_num + EPISILON) 
		self.tracking_loc_loss = self.tracking_loc_loss.view(1)
		if self.training:
			rpn_loss_cls.unsqueeze_(0)
			rpn_loss_bbox.unsqueeze_(0)
			self.RFCN_loss_cls.unsqueeze_(0)
			self.RFCN_loss_bbox.unsqueeze_(0)

		return  output, self.tracking_cls_loss.view(1), \
				self.tracking_loc_loss.view(1), \
				self.tracking_predict_loc_loss.view(1), \
				rpn_loss_cls, rpn_loss_bbox, \
				self.RFCN_loss_cls, self.RFCN_loss_bbox

