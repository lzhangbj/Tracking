# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
def get_minibatch(roidb, num_classes):
	"""Given a roidb, construct a minibatch sampled from it."""
	'''
		due to rpn, the batch size can only be 1
		a im_blob is an array containing a batch of rois
		
		input:
		@roidb: [roidb], since batch size is 1
		
		output:
		@blobs: a dict 
		{
			'data'    : imblob of shape (batchsize, max height in batch of imgs, max width in batch of imgs, 3)
			'gt_boxes': list of boxes[(x1, y1, x2, y2, class_index)]
			'im_info' : a batch of [height, width, scale], the width and height are scaled, scale is the scaled value
			'im_id'   : id in [0, imdb.num_images)
		}

	'''
	num_images = len(roidb)
	# Sample random scales to use for each image in this batch
	random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
									size=num_images)
	assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
		'num_images ({}) must divide BATCH_SIZE ({})'. \
		format(num_images, cfg.TRAIN.BATCH_SIZE)

	# Get the input image blob, formatted for caffe
	im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

	blobs = {'data': im_blob}

	assert len(im_scales) == 1, "Single batch only"
	assert len(roidb) == 1, "Single batch only"
	
	# gt boxes: (x1, y1, x2, y2, cls)
	if cfg.TRAIN.USE_ALL_GT:
		# Include all ground truth boxes
		gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
	else:
		# For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
		gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
	gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
	gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
	gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
	blobs['gt_boxes'] = gt_boxes
	blobs['im_info'] = np.array(
		[[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
		dtype=np.float32)

	blobs['img_id'] = roidb[0]['img_id']
	return blobs

def get_vid_minibatch(roidb, num_classes):
	"""Given a roidb, construct a minibatch sampled from it."""
	'''
		due to rpn, the batch size can only be 
		a im_blob is an array containing a batch of rois
		
		input:
		@roidb: [roidb]*seq_length, since batch size is  1
		
		output:
		@blobs: a dict 
		{
			'data'    : imblob of shape (batchsize, max height in batch of imgs, max width in batch of imgs, 3)      	np array
			'gt_boxes': a list of video length, each component if a list of boxes[(x1, y1, x2, y2, class_index)] 		python list
			'im_info' : a list of [height, width, scale], the width and height are scaled, scale is the scaled value 	np array
						since the im_info in a video are the same, we limit list lenght to 1 
			'im_id'   : a list of id in [0, imdb.num_images) 															np array
		}

	'''
	num_videos = 1
	# Sample random scales to use for each image in this batch
	random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES))
	assert(cfg.TRAIN.BATCH_SIZE % num_videos == 0), \
		'num_videos ({}) must divide BATCH_SIZE ({})'. \
		format(num_videos, cfg.TRAIN.BATCH_SIZE)

	# Get the input image blob, formatted for caffe
	# im_blob (seq_length, h, w, channel)
	# in_csacles :scalar
	im_blob, im_scale = _get_vid_image_blob(roidb, random_scale_inds) 


	blobs = {'data': im_blob}
	blobs['gt_boxes'] = []
	blobs['img_id']   = []
	# assert len(im_scales) == 1, "one video has only one scale"
	
	for i in range(len(roidb)):
		# gt boxes: (x1, y1, x2, y2, cls)
		if cfg.TRAIN.USE_ALL_GT:
			# Include all ground truth boxes
			gt_inds = np.where(roidb[i]['gt_classes'] != 0)[0]
		else:
			# For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
			gt_inds = np.where((roidb[i]['gt_classes'] != 0) & np.all(roidb[i]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
		gt_boxes = np.empty((len(gt_inds), 6), dtype=np.float32)
		gt_boxes[:, :5] = roidb[i]['boxes'][gt_inds, :5] * (np.array([im_scale]*4 + [1]).reshape(1,5))
		# uvs = np.unique(gt_boxes[:, 4])
		# assert uvs.shape[0] == gt_boxes.shape[0], gt_boxes[:, 4]
		# gt_boxes[:, 5] = roidb[i]['boxes'][gt_inds, 4]
		gt_boxes[:, 5] = roidb[i]['gt_classes'][gt_inds]
		blobs['gt_boxes'].append(gt_boxes)
		blobs['img_id'].append(roidb[i]['img_id'])

	# blobs['gt_boxes'] = np.array(blobs['gt_boxes'], dtype=np.float32)  cant be np array since gt boxes num diffes every frame
	# blobs['gt_boxes'] = blobs['gt_boxes']
	blobs['im_info'] = np.array(
		[im_blob.shape[1], im_blob.shape[2], im_scale], # same as get minibatch
		dtype=np.float32)
	blobs['img_id'] = np.array(blobs['img_id'], dtype=np.float32)

	return blobs

def _get_image_blob(roidb, scale_inds):
	"""Builds an input blob from the images in the roidb at the specified
	scales.
	"""
	num_images = len(roidb)

	processed_ims = []
	im_scales = []
	for i in range(num_images):
		#im = cv2.imread(roidb[i]['image'])
		im = imread(roidb[i]['image'])

		if len(im.shape) == 2:
			im = im[:,:,np.newaxis]
			im = np.concatenate((im,im,im), axis=2)
		# flip the channel, since the original one using cv2
		# rgb -> bgr
		im = im[:,:,::-1]

		if roidb[i]['flipped']:
			im = im[:, ::-1, :]
		target_size = cfg.TRAIN.SCALES[scale_inds[i]]
		im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
										cfg.TRAIN.MAX_SIZE)
		im_scales.append(im_scale)
		processed_ims.append(im)

	# Create a blob to hold the input images
	blob = im_list_to_blob(processed_ims)

	return blob, im_scales

def _get_vid_image_blob(roidb, scale_inds):
	"""Builds an input blob from the images in the roidb at the specified
	scales.
	"""

	processed_ims = []
	# im_scales = []
	for i in range(len(roidb)):
		im = imread(roidb[i]['image'])
		if len(im.shape) == 2:
			im = im[:,:,np.newaxis]
			im = np.concatenate((im,im,im), axis=2)
		# flip the channel, since the original one using cv2
		# rgb -> bgr
		im = im[:,:,::-1]

		if roidb[i]['flipped']:
			im = im[:, ::-1, :]
		target_size = cfg.TRAIN.SCALES[scale_inds]
		im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
										cfg.TRAIN.MAX_SIZE)
		processed_ims.append(im)
	# im_scales.append(im_scale)
	# Create a blob to hold the input images
	blob = im_list_to_blob(processed_ims) # (seq_lenght, h, w, channel)

	return blob, im_scale

