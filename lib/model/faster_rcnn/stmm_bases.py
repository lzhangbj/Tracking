import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
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


class STMM_Cell(nn.Module):
	"""
	Generate a convolutional GRU cell
	"""

	def __init__(self, in_channels=2048, out_channels=1024):
		'''
		settings of W are the same as conv_new_1 in rfcn
		'''
		super(STMM_Cell, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		kernel_size=1
		stride = 1
		self.k=2
		self.Wz = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
		self.Wr = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)
		self.W = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)        

		self.Uz = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=0)
		self.Ur = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=0)
		self.U = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=0)

		# for matchtrans
		self.sum_filter = nn.Conv2d(2048, 2048, kernel_size=2*self.k+1, stride=1, padding=self.k, bias=False) 
		self.sum_filter.weight.requires_grad = False
		self.sum_filter.weight.data.fill_(1.)       

	def W_pretrained_init(self, weights, bias):
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
		mu_X = torch.mean(input, dim=0).unsqueeze_(0) # adding batch dimension for broadcasting in comparing with input
		std_X = torch.std(input, dim=0).unsqueeze_(0)
		X_limit = mu_X + K*std_X
		input[input>X_limit]=1
		return input

	def _MatchTrans(self, curr_F, prev_F, prev_M):
		'''
		@F: (batch, 2048, h, w)
		@M: (batch, 1024, h, w)
		'''
		bs, channel, h, w = prev_F.size()
		k = self.k

		prev_F_no_c = torch.sum(prev_F, dim=1)
		curr_F_no_c = torch.sum(curr_F, dim=1)
		prev_F_sum = self.sum_filter(prev_F)
		prev_M_exp = prev_M.unsqueeze(4).unsqueeze(5)

		mask_denominator = curr_F_no_c * torch.sum(prev_F_sum, dim=1)
		mask = torch.zeros([bs, 1, h, w, 2*k+1, 2*k+1], dtype=torch.float32).cuda()
		aligned_prev_M = torch.zeros_like(prev_M_exp, dtype=torch.float32).cuda()
		for i in range(-k, k+1):
			for j in range(-k, k+1):
				moved_prev_F = torch.zeros_like(prev_F_no_c, dtype=torch.float32)
				moved_prev_F[:, max(0, i):min(h+i, h), max(0, j):min(w+j, w)] = \
						prev_F_no_c[:, max(-i, 0):min(h, h-i), max(-j, 0):min(w, w-j)]
				mask[:,:, :,  :, i+k, j+k] = curr_F_no_c*moved_prev_F / mask_denominator
				assert mask[:, :, max(0, i):min(h+i, h), max(0, j):min(w+j, w), i+k, j+k].size()[2:] == prev_M_exp[:, :, max(-i, 0):min(h, h-i), max(-j, 0):min(w, w-j), :, :].size()[2:], \
						[mask[:, :, max(0, i):min(h+i, h), max(0, j):min(w+j, w), i+k, j+k].size(), prev_M_exp[:, :, max(-i, 0):min(h, h-i), max(-j, 0):min(w, w-j), :, :].size()]
				aligned_prev_M += mask[:, :, max(0, i):min(h+i, h), max(0, j):min(w+j, w), i+k, j+k]\
								*prev_M_exp[:, :, max(-i, 0):min(h, h-i), max(-j, 0):min(w, w-j), :, :]
		aligned_prev_M = torch.sum(aligned_prev_M, (4,5))
		return aligned_prev_M

	def forward(self, curr_F, prev_F, prev_M, body=1):
		'''
		input && return
		( batch, C, H, W )
		body =0:head
		body!=0:body 
		'''
		# get batch and spatial sizes
		assert curr_F.size()[1] == 2048, curr_F.size()

		if body:
			aligned_prev_M = self._MatchTrans(curr_F, prev_F, prev_M)
		else:
			aligned_prev_M = prev_M
		zt = self._BatchNormStar(\
			F.relu(\
				self.Wz(curr_F) + self.Uz(aligned_prev_M)
				)
			)
		rt = self._BatchNormStar(\
			F.relu(\
				self.Wr(curr_F) + self.Ur(aligned_prev_M)
				)
			)
		M = F.relu(\
				self.W(curr_F) + self.U(aligned_prev_M*rt)
				)
		curr_M = (1-zt)*aligned_prev_M + zt*M
		assert curr_M.size(1) == 1024, curr_M.size()

		return curr_M


class STMM_RFCN(nn.Module):
	""" faster RCNN """
	def __init__(self, classes, class_agnostic):
		super(STMM_RFCN, self).__init__()
		self.k = 7
		self.classes = classes
		self.n_classes = len(classes)
		self.class_agnostic = class_agnostic

		self.rfcn_cls = nn.Conv2d(in_channels=1024, out_channels=self.n_classes * self.k * self.k,
								kernel_size=1, stride=1, padding=0, bias=False)
		self.rfcn_bbox = nn.Conv2d(in_channels=1024, out_channels= self.n_classes*4 * self.k * self.k,
								kernel_size=1, stride=1, padding=0, bias=False) #--class-agnostic
		self.psroipooling_cls = PSRoIPool(pooled_height=self.k, pooled_width=self.k,
										spatial_scale= 1/32.0, group_size=self.k, 
										output_dim=self.n_classes)
		self.psroipooling_loc = PSRoIPool(pooled_height=self.k, pooled_width=self.k,
										spatial_scale= 1/32.0,   group_size=self.k,
										output_dim= self.n_classes * 4) #----class-agnostic

		self.pooling = nn.AvgPool2d(kernel_size=self.k, stride=self.k)

	def forward(self, feat, rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws):
		'''
		feat:       (batch, c, h, w) c should be 1024
		rois:       (batch, max_num, 5)
		rois_label: (batch*max_num)
		'''
		batch_size = feat.size(0)

		rfcn_cls = self.rfcn_cls(feat) # output (batch, k*k*n_classes, h, w)
		rfcn_bbox = self.rfcn_bbox(feat)  # output (batch, 8*k*k, h, w) 
		#----cls
		psroipooled_cls_rois = self.psroipooling_cls(rfcn_cls, rois.view(-1, 5)) # shape (num_rois,  n_classes, k, k)
		ave_cls_score_rois = self.pooling(psroipooled_cls_rois) # vote, ave the k*k psmap
		ave_cls_score_rois = ave_cls_score_rois.squeeze() # shape (batch,  )
		cls_score_pred = F.softmax(ave_cls_score_rois, dim=1) #(n_rois, n_classes)
		#---loc
		psroipooled_loc_rois = self.psroipooling_loc(rfcn_bbox, rois.view(-1, 5)) # shape (num_rois, 4*n_classes, k, k)
		ave_bbox_pred_rois = self.pooling(psroipooled_loc_rois) 
		
		if self.training and not self.class_agnostic:    #
			# select the corresponding columns according to roi labels
			bbox_pred_view = ave_bbox_pred_rois.view(ave_bbox_pred_rois.size(0), int(ave_bbox_pred_rois.size(1) / 4), 4)
			# shape (n_rois, n_classes, 4)
			bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
			ave_bbox_pred_rois = bbox_pred_select.squeeze(1)

		if self.training:
			# classification loss
			RCNN_loss_cls = F.cross_entropy(ave_cls_score_rois, rois_label)
			# bounding box regression L1 loss
			RCNN_loss_bbox = _smooth_l1_loss(ave_bbox_pred_rois, rois_target, rois_inside_ws, rois_outside_ws)

		cls_prob = cls_score_pred.view(batch_size, rois.size(1), -1)
		bbox_pred = ave_bbox_pred_rois.view(batch_size, rois.size(1), -1)

		return cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox

	# def _init_weights(self):
	#     def normal_init(m, mean, stddev, truncated=False):
	#         """
	#         weight initalizer: truncated normal and random normal.
	#         """
	#         # x is a parameter
	#         if truncated:
	#             m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
	#         else:
	#             m.weight.data.normal_(mean, stddev)
	#             m.bias.data.zero_()

	#     normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
	#     normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
	#     normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
	#     normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
	#     normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

	# def create_architecture(self):
	#     self._init_modules()
	#     self._init_weights()
