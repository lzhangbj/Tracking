
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_vid_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class vidbatchLoader(data.Dataset):
	def __init__(self, vid_structure, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
		self._vid_structure = vid_structure
		self._roidb = roidb
		self._num_classes = num_classes
		# we make the height of image consistent to trim_height, trim_width
		self.trim_height = cfg.TRAIN.TRIM_HEIGHT
		self.trim_width = cfg.TRAIN.TRIM_WIDTH
		self.max_num_box = cfg.MAX_NUM_GT_BOXES
		self.training = training
		self.normalize = normalize
		self.ratio_list = ratio_list  # ratio list is a list of width/height needs crop if ratio > 2 or < 0.5
		self.ratio_index = ratio_index
		self.batch_size = batch_size
		self.data_size = len(self.ratio_list)
		self.gt_boxes_id = []

		# given the ratio_list, we want to make the ratio same for each batch.
		self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
		num_batch = int(np.ceil(len(ratio_index) / batch_size))
		for i in range(num_batch):
			left_idx = i*batch_size
			right_idx = min((i+1)*batch_size-1, self.data_size-1) 

			if ratio_list[right_idx] < 1:
				# for ratio < 1, we preserve the leftmost in each batch.
				target_ratio = ratio_list[left_idx]
			elif ratio_list[left_idx] > 1:
				# for ratio > 1, we preserve the rightmost in each batch.
				target_ratio = ratio_list[right_idx]
			else:
				# for ratio cross 1, we make it to be 1.
				target_ratio = 1

			self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

	def __getitem__(self, index): # the index th  video
		'''
		return 
		padding_data_list 	: (video_len, 3, h, w)
		im_info				: (h, w, scale)
		gt_boxes_padding 	: (video_len, max_num_boxes, 5)
		num_boxes 			: (video_len, 1)    
		these values are directly return for each batch when training
		
		in training we set video len to 7, while test is 11, this needs to be set in vid_structure, which genereted from data files 
		'''
		if self.training:
			index_ratio = int(self.ratio_index[index])
		else:
			index_ratio = index

		# get the anchor index for current sample index
		# here we set the anchor index to the last one
		# sample in this group
		minibatch_db = self._roidb[index_ratio] # a minibatch is a batch of frames(one video)
		blobs = get_vid_minibatch(minibatch_db, self._num_classes) # one video ()
		# print(blobs['img_id'])
		data = torch.from_numpy(blobs['data']) #(seq_len, h, w, chanel)
		gt_boxes = [torch.from_numpy(blobs['gt_boxes'][i]) for i in range(len(blobs['gt_boxes']))]
		im_info = torch.from_numpy(blobs['im_info'])  # (h,w,scale) tuple
		# we need to random shuffle the bounding box.
		data_height, data_width = data.size(1), data.size(2)
		# if the image need to crop, crop to the target size.
		ratio = self.ratio_list_batch[index] # target ratio

		if self.training:
			gt_boxes_padding_list = []
			num_boxes_list = []
			########################################################
			# padding the input image to fixed size for each group #
			########################################################

			# NOTE1: need to cope with the case where a group cover both conditions. (done)
			# NOTE2: need to consider the situation for the tail samples. (no worry)
			# NOTE3: need to implement a parallel data loader. (no worry)
			# get the index range
			if self._roidb[index_ratio][0]['need_crop']:
				if ratio < 1:
					# this means that data_width << data_height, in our case, 0.5,  we need to crop the
					# data_height
					trim_size= int(np.floor(data_width / ratio))
					if trim_size > data_height:
						trim_size = data_height  
					min_y = 10000
					max_y = 0
					for i in range(len(data)):
						np.random.shuffle(blobs['gt_boxes'][i])
						gt_box = gt_boxes[i]
						min_y = min(int(torch.min(gt_box[:,1])), min_y)
						max_y = max(int(torch.max(gt_box[:,3])), max_y)
					box_region = max_y - min_y + 1
					if min_y == 0:
						y_s = 0
					else:
						if (box_region-trim_size) < 0:
							y_s_min = max(max_y-trim_size, 0)
							y_s_max = min(min_y, data_height-trim_size)
							if y_s_min == y_s_max:
								y_s = y_s_min
							else:
								y_s = np.random.choice(range(y_s_min, y_s_max))
						else:
							y_s_add = int((box_region-trim_size)/2)
							if y_s_add == 0:
								y_s = min_y
							else:
								y_s = np.random.choice(range(min_y, min_y+y_s_add))
					# crop the image
					# if no boxes found
					data = data[:, y_s:(y_s + trim_size), :, :]
					# shift y coordiante of gt_boxes
					for kk in range(len(gt_boxes)):
						gt_boxes[kk][:, 1] = gt_boxes[kk][:, 1] - float(y_s)
						gt_boxes[kk][:, 3] = gt_boxes[kk][:, 3] - float(y_s)

						# update gt bounding box according the trip
						gt_boxes[kk][:, 1].clamp_(0, trim_size - 1)
						gt_boxes[kk][:, 3].clamp_(0, trim_size - 1)
						
				else:
					# this means that data_width >> data_height, we need to crop the
					# data_width
					trim_size = int(np.ceil(data_height * ratio))
					if trim_size > data_width:
						trim_size = data_width 
					min_x = 10000
					max_x = 0
					for i in range(len(data)):
						np.random.shuffle(blobs['gt_boxes'][i])
						gt_box = gt_boxes[i]
						min_x = min(int(torch.min(gt_box[:,0])), min_x)
						max_x = max(int(torch.max(gt_box[:,2])), max_x)
					box_region = max_x - min_x + 1
					if min_x == 0:
						x_s = 0
					else:
						if (box_region-trim_size) < 0:
							x_s_min = max(max_x-trim_size, 0)
							x_s_max = min(min_x, data_width-trim_size)
							if x_s_min == x_s_max:
								x_s = x_s_min
							else:
								x_s = np.random.choice(range(x_s_min, x_s_max))
						else:
							x_s_add = int((box_region-trim_size)/2)
							if x_s_add == 0:
								x_s = min_x
							else:
								x_s = np.random.choice(range(min_x, min_x+x_s_add))
					# crop the image
					data = data[:, :, x_s:(x_s + trim_size), :]
					for kk in range(len(gt_boxes)):
						# shift x coordiante of gt_boxes
						gt_boxes[kk][:, 0] = gt_boxes[kk][:, 0] - float(x_s)
						gt_boxes[kk][:, 2] = gt_boxes[kk][:, 2] - float(x_s)
						# update gt bounding box according the trip
						gt_boxes[kk][:, 0].clamp_(0, trim_size - 1)
						gt_boxes[kk][:, 2].clamp_(0, trim_size - 1)

			# based on the ratio, padding the image.
			if ratio < 1:
				# this means that data_width < data_height
				trim_size = int(np.floor(data_width / ratio))

				padding_data = torch.FloatTensor(len(data), int(np.ceil(data_width / ratio)), \
												 data_width, 3).zero_()

				padding_data[:, :data_height, :, :] = data
				# update im_info i.e. h
				im_info[0] = padding_data.size(1) 
				# print("height %d %d \n" %(index, anchor_idx))
			elif ratio > 1:
				# this means that data_width > data_height
				# if the image need to crop.
				padding_data = torch.FloatTensor(len(data), data_height, \
												 int(np.ceil(data_height * ratio)), 3).zero_()
				padding_data[:, :, :data_width, :] = data
				im_info[1] = padding_data.size(2)
			else:
				trim_size = min(data_height, data_width)
				padding_data = torch.FloatTensor(len(data), trim_size, trim_size, 3).zero_()
				padding_data = data[:, :trim_size, :trim_size, :]
				# gt_boxes.clamp_(0, trim_size)
				for kk in range(len(gt_boxes)):
					gt_boxes[kk][:, :4].clamp_(0, trim_size)
				im_info[0] = trim_size
				im_info[1] = trim_size

			# check the bounding box:
			for i in range(len(data)):
				gt_box = gt_boxes[i]
				not_keep = (gt_box[:,0] == gt_box[:,2]) | (gt_box[:,1] == gt_box[:,3])
				keep = torch.nonzero(not_keep == 0).view(-1)

				gt_box_padding = torch.FloatTensor(1,self.max_num_box, gt_box.size(1)).zero_()
				gt_box_padding[:, :, 4] = -1
				if keep.numel() != 0:
					gt_box = gt_box[keep]
					num_boxes = np.minimum(gt_box.size(0), self.max_num_box)
					gt_box_padding[:, :num_boxes,:] = gt_box[:num_boxes]
					uvs = torch.unique(gt_box_padding[:, :num_boxes,4])
					assert uvs.size(0) == num_boxes, gt_box_padding[:, :num_boxes,4]
				else:
					num_boxes = 0
				gt_boxes_padding_list.append(gt_box_padding)
				num_boxes_list.append(num_boxes)
				# permute trim_data to adapt to downstream processing
			padding_data = padding_data.permute(0, 3, 1, 2).contiguous()
			im_info = im_info.view(3)
			gt_boxes_padding_list = torch.cat(gt_boxes_padding_list)
			num_boxes_list = torch.FloatTensor(num_boxes_list)
			return padding_data, im_info, gt_boxes_padding_list, num_boxes_list
		else:
			data 			= data.permute(0, 3, 1, 2).contiguous().view(-1, 3, data_height, data_width)
			im_info 		= im_info.view(3)
			gt_boxes_list 	=  torch.FloatTensor([[1,1,1,1,-1,1]]*len(data))
			num_boxes = torch.FloatTensor([0]*len(data))
			return data, im_info, gt_boxes_list, num_boxes

	def __len__(self):
		return len(self.ratio_list)
