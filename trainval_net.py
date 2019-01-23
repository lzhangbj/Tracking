# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
'''
	things to do:
	tensorboard
	savemodel
	look faster rcnn model
	maybe look imdb
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.vidbatchLoader import vidbatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
			adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet, resnet_rfcn
from model.faster_rcnn.stmm import STMM
from model.faster_rcnn.tracking_cell import TrackingCell

# from model.utils.my_data_parallel import DataParallel

import psutil
import gc
import torch.cuda as cutorch

from subprocess import call

import smtplib
 
# def email_alert(msg):
# 	SERVER = smtplib.SMTP('smtp.gmail.com', 25)
# 	SERVER.starttls()
# 	SERVER.login("1999forrestz@gmail.com", "990218Googleaccount")
	
# 	server.sendmail("1999forrestz@gmail.com", "1999forrestz@gmail.com", msg)
# 	server.quit()
def output_alert(msg):
	with open("{}.txt".format(msg), 'wb') as f:
		f.write("ok now")

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

def cpuStats():
		pid = os.getpid()
		py = psutil.Process(pid)
		memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
		print('memory GB:', memoryUse)

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('--dataset', dest='dataset',
											help='training dataset',
											default='imagenet_vid', type=str)
	parser.add_argument('--model', dest='model',
											help='model to use',
											default='track', type=str)
	parser.add_argument('--video', dest='video',
											help='if using video mode or not',
											action='store_true')    
	parser.add_argument('--net', dest='net',
										help='vgg16, res101',
										default='res101', type=str)
	parser.add_argument('--start_epoch', dest='start_epoch',
											help='starting epoch',
											default=1, type=int)
	parser.add_argument('--epochs', dest='max_epochs',
											help='number of epochs to train',
											default=20, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
											help='number of iterations to display',
											default=100, type=int)
	parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
											help='number of iterations to display',
											default=10000, type=int)

	parser.add_argument('--save_dir', dest='save_dir',
											help='directory to save models', default="models",
											type=str)
	parser.add_argument('--nw', dest='num_workers',
											help='number of worker to load data',
											default=0, type=int)
	parser.add_argument('--cuda', dest='cuda',
											help='whether use CUDA',
											action='store_true')
	parser.add_argument('--ls', dest='large_scale',
											help='whether use large imag scale',
											action='store_true')                      
	parser.add_argument('--mGPUs', dest='mGPUs',
											help='whether use multiple GPUs',
											action='store_true')
	parser.add_argument('--bs', dest='batch_size',
											help='batch_size',
											default=1, type=int)
	parser.add_argument('--cag', dest='class_agnostic',
											help='whether perform class_agnostic bbox regression',
											action='store_true')

# config optimization
	parser.add_argument('--o', dest='optimizer',
											help='training optimizer',
											default="adam", type=str)
	parser.add_argument('--lr', dest='lr',
											help='starting learning rate',
											default=0.001, type=float)
	parser.add_argument('--lr_decay_step', dest='lr_decay_step',
											help='step to do learning rate decay, unit is epoch',
											default=5, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
											help='learning rate decay ratio',
											default=0.1, type=float)

# set training session
	parser.add_argument('--s', dest='session',
											help='training session',
											default=1, type=int)

# resume trained model
	parser.add_argument('--r', dest='resume',
											help='resume checkpoint or not',
											default=False, type=bool)
	parser.add_argument('--checksession', dest='checksession',
											help='checksession to load model',
											default=1, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
											help='checkepoch to load model',
											default=1, type=int)
	parser.add_argument('--checkpoint', dest='checkpoint',
											help='checkpoint to load model',
											default=0, type=int)
# log and diaplay
	parser.add_argument('--use_tfb', dest='use_tfboard',
											help='whether use tensorboard',
											action='store_true')

	args = parser.parse_args()
	return args


class sampler(Sampler):
	def __init__(self, train_size, batch_size):
		self.num_data = train_size
		self.num_per_batch = int(train_size / batch_size) 
		# num per batch is actually the num of batches
		self.batch_size = batch_size
		self.range = torch.arange(0,batch_size).view(1, batch_size).long()
		self.leftover_flag = False
		if train_size % batch_size:
			self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
			self.leftover_flag = True

	def __iter__(self):
		rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
		# rand_num = torch.arange(self.num_per_batch).view(-1,1) * self.batch_size
		self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

		self.rand_num_view = self.rand_num.view(-1) # only shuffle the batches

		if self.leftover_flag:
			self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

		return iter(self.rand_num_view)

	def __len__(self):
		return self.num_data

if __name__ == '__main__':

	args = parse_args()

	print('Called with args:')
	print(args)

	if args.dataset == "pascal_voc":
		args.imdb_name = "voc_2007_trainval"
		args.imdbval_name = "voc_2007_test"
		args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "pascal_voc_0712":
		args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
		args.imdbval_name = "voc_2007_test"
		args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "coco":
		args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
		args.imdbval_name = "coco_2014_minival"
		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
	elif args.dataset == "imagenet":
		args.imdb_name = "imagenet_train"
		args.imdbval_name = "imagenet_val"
		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
	elif args.dataset == "vg":
		# train sizes: train, smalltrain, minitrain
		# train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
		args.imdb_name = "vg_150-50-50_minitrain"
		args.imdbval_name = "vg_150-50-50_minival"
		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
	elif args.dataset == "imagenet_vid":
		args.imdb_name = "vid_2015_train"
		args.imdbval_name = "vid_2015_test"  # useless now
		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
	elif args.dataset == "imagenet_vid_img":
		args.imdb_name = "vid_img_2015_train"
		args.imdbval_name = "vid_img_2015_val"
		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
	args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	#torch.backends.cudnn.benchmark = True
	if torch.cuda.is_available() and not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# train set
	# -- Note: Use validation set and disable the flipped to enable faster loading.
	cfg.TRAIN.USE_FLIPPED = True
	cfg.USE_GPU_NMS = args.cuda
	# for images, the ratio_list and ratio_index has the same length as the roidb, i.e. # of images
	# for videos, the ratio_list and ratio_index has the same length as the videos, i.e. # of videos
	imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
	with open("roidb_loded.txt", 'wb') as f:
		pass
	print("roidb created")
	sys.stdout.flush()

	if not args.video: ###########################################################################################
		train_size = len(roidb)
	else:
		train_size = len(roidb)

	print('{:d} roidb entries'.format(len(roidb)))

	output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	sampler_batch = sampler(train_size, args.batch_size)

	if not args.video:
		dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
													 imdb.num_classes, training=True)
	else:
		dataset = vidbatchLoader(imdb._video_structure, roidb, ratio_list, ratio_index, args.batch_size, \
													 imdb.num_classes, training=True)
	# if not args.video:
	# 	print(dataset.compute_mean())
	print("dataset created")
	sys.stdout.flush()
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
														sampler=sampler_batch, num_workers=args.num_workers) 
														# zl: dataset [0,1,2,3] corresponds to imdata, iminfo, num_boxes, gt_boxes
	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)

	# ship to cuda
	if args.cuda:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if args.cuda:
		cfg.CUDA = True

	pretrained_rfcn = True;

	# initilize the network here.
	if args.net == 'vgg16':
		fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res101':
		if args.model == "faster_rcnn":
			fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)        
		elif args.model == "rfcn":
			fasterRCNN = resnet_rfcn(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
		elif args.model == "stmm":
			fasterRCNN = STMM(imdb.classes, rfcn_pretrained=False, class_agnostic=args.class_agnostic)
		elif args.model == "track":
			fasterRCNN = TrackingCell(imdb.classes, class_agnostic=args.class_agnostic,pretrained_rfcn="rfcn_detect.pth") #"models/res101/imagenet_vid_img/rfcn_2_10_29197.pth"
	elif args.net == 'res50':
		fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res152':
		fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
	else:
		print("network is not defined")
		pdb.set_trace() 
	print("architecture creating")
	sys.stdout.flush()
	fasterRCNN.create_architecture() # init modules and weights
	print("architecture created")
	sys.stdout.flush()
	lr = cfg.TRAIN.LEARNING_RATE
	lr = args.lr
	#tr_momentum = cfg.TRAIN.MOMENTUM
	#tr_momentum = args.momentum

	params = []
	for key, value in dict(fasterRCNN.named_parameters()).items():
		if value.requires_grad:
			if 'bias' in key:
				if False and args.video and 'RCNN' in key or "rfcn" in key or "conv_new_1" in key:
					params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1)*0.1, \
								'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
				else:
					params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
								'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
			else:
				if False and args.video and 'RCNN' in key or "rfcn" in key or "conv_new_1" in key:
					params += [{'params':[value],'lr':lr*0.1, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
				else:
					params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
			
				

	if args.optimizer == "adam":
		lr = lr * 0.1
		optimizer = torch.optim.Adam(params)

	elif args.optimizer == "sgd":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	if args.cuda:
		fasterRCNN.cuda() ################################################
		# if args.video:
		# 	fasterRCNN.cuda_transform()
 
	if args.resume:
		# load_name = os.path.join(output_dir,
		# 	'rfcn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
		load_name = os.path.join("rfcn_detect.pth")
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		args.session = checkpoint['session']
		args.start_epoch = checkpoint['epoch']
		fasterRCNN.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])  
		lr = optimizer.param_groups[0]['lr']
		if 'pooling_mode' in checkpoint.keys():
			cfg.POOLING_MODE = checkpoint['pooling_mode']
		print("loaded checkpoint %s" % (load_name))
		sys.stdout.flush()

	if args.mGPUs:
		fasterRCNN = nn.DataParallel(fasterRCNN)

	iters_per_epoch = int(train_size / args.batch_size) # ZL: what about leftover

	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		logger = SummaryWriter("logs")

	# msg = "vid_track_gt_roidb_finish"
	# output_alert(msg)	

	# for o,i in enumerate(roidb):
	# 	if i['img_id'] > 879180 and i['img_id'] < 879185:
	# 		print(imdb._image_index[o])
	# 		print(i)

	# test the roidb loader
	# data_iter = iter(dataloader)
	# for step in range(iters_per_epoch):
	# 	sys.stdout.write("\r<<<<< loading {:d}/{:d} id: ".format(step, iters_per_epoch))
	# 	sys.stdout.flush()
	# 	data = next(data_iter)

	


	for epoch in range(args.start_epoch, args.max_epochs + 1):
		# setting to train mode
		fasterRCNN.train()
		loss_temp = 0
		start = time.time()

		if epoch % (args.lr_decay_step + 1) == 0:
				adjust_learning_rate(optimizer, args.lr_decay_gamma) # zl decay ALL learning rate, lr times decay_gamma
				lr *= args.lr_decay_gamma

		data_iter = iter(dataloader)
		for step in range(iters_per_epoch): # one step is one batch
			data = next(data_iter)
			im_data.data.resize_(data[0].size()).copy_(data[0])
			im_info.data.resize_(data[1].size()).copy_(data[1])
			gt_boxes.data.resize_(data[2].size()).copy_(data[2])
			num_boxes.data.resize_(data[3].size()).copy_(data[3])

			input = None
			# call(["nvidia-smi"])
			fasterRCNN.zero_grad()
			if args.model == 'track':
				vid_len = 6
				for i in [0]:
					
					im_data_fold = im_data[:, i:i+6, :, :, :]
					im_info_fold = im_info
					gt_boxes_fold = gt_boxes[:, i:i+6, :, :]
					num_boxes_fold = num_boxes[:, i:i+6]				

					# track_rois, track_ids, track_cls, track_num, \
					input, track_cls_loss, track_loc_loss, track_predict_loc_loss, \
					rpn_loss_cls, rpn_loss_box, \
					RCNN_loss_cls, RCNN_loss_bbox = fasterRCNN(im_data_fold,\
																	 im_info_fold, \
																	 gt_boxes_fold,\
																	 num_boxes_fold, input) # zl forward		
					loss =  track_cls_loss.mean() + track_loc_loss.mean() + track_predict_loc_loss.mean() 
							#track_predict_loc_loss.mean() 
							# +\
							 # rpn_loss_cls.mean() + rpn_loss_box.mean() \
							 # + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
					loss_temp += float(loss.item())
					# backward
					optimizer.zero_grad()
					loss.backward()
					# if args.net == "vgg16":
					# clip_gradient(fasterRCNN, 1.) # error prone
					optimizer.step()

					input = [p.detach() for p in input]
					# input, track_cls_loss, track_loc_loss, track_predict_loc_loss, \
					# rpn_loss_cls, rpn_loss_box, \
					# RCNN_loss_cls, RCNN_loss_bbox = fasterRCNN(im_data2,\
					# 												 im_info2, \
					# 												 gt_boxes2,\
					# 												 num_boxes2, input) # zl forward		
					# loss = track_cls_loss.mean() + track_loc_loss.mean() + track_predict_loc_loss.mean() \
					# 		+ rpn_loss_cls.mean() + rpn_loss_box.mean() \
					# 		 + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()	

					

				# gpuStats()

				# device = torch.device('cuda:1')
				# input = [t.cuda(1) for t in input ]
				# fasterRCNN = fasterRCNN.to(device)
				# input, track_rois_, track_ids_, track_cls_, track_num_, \
				# track_cls_loss_, track_loc_loss_, track_predict_loc_loss_, \
				# rpn_loss_cls_, rpn_loss_box_, \
				# RCNN_loss_cls_, RCNN_loss_bbox_ = fasterRCNN(im_data[:,2:4,:,:,:],\
				# 												 im_info, \
				# 												 gt_boxes[:,2:4,:,:],\
				# 												 num_boxes[:, 2:4], input) # zl forward	
				# track_cls_loss = torch.cat([track_cls_loss,track_cls_loss_])
				# track_loc_loss = torch.cat([track_loc_loss,track_loc_loss_])
				# track_predict_loc_loss = torch.cat([track_predict_loc_loss,track_predict_loc_loss_])
				# rpn_loss_cls   = torch.cat([rpn_loss_cls,rpn_loss_cls_])
				# rpn_loss_box   = torch.cat([rpn_loss_box,rpn_loss_box_])
				# RCNN_loss_cls  = torch.cat([RCNN_loss_cls,RCNN_loss_cls_])
				# RCNN_loss_bbox = torch.cat([RCNN_loss_bbox,RCNN_loss_bbox_])

				# input, track_rois_, track_ids_, track_cls_, track_num_, \
				# track_cls_loss_, track_loc_loss_, track_predict_loc_loss_, \
				# rpn_loss_cls_, rpn_loss_box_, \
				# RCNN_loss_cls_, RCNN_loss_bbox_ = fasterRCNN(im_data[:,4:6,:,:,:],\
				# 												 im_info, \
				# 												 gt_boxes[:,4:6,:,:],\
				# 												 num_boxes[:, 4:6], input) # zl forward	
				# track_cls_loss = torch.cat([track_cls_loss,track_cls_loss_])
				# track_loc_loss = torch.cat([track_loc_loss,track_loc_loss_])
				# track_predict_loc_loss = torch.cat([track_predict_loc_loss,track_predict_loc_loss_])
				# rpn_loss_cls   = torch.cat([rpn_loss_cls,rpn_loss_cls_])
				# rpn_loss_box   = torch.cat([rpn_loss_box,rpn_loss_box_])
				# RCNN_loss_cls  = torch.cat([RCNN_loss_cls,RCNN_loss_cls_])
				# RCNN_loss_bbox = torch.cat([RCNN_loss_bbox,RCNN_loss_bbox_])

				# input, track_rois_, track_ids_, track_cls_, track_num_, \
				# track_cls_loss_, track_loc_loss_, track_predict_loc_loss_, \
				# rpn_loss_cls_, rpn_loss_box_, \
				# RCNN_loss_cls_, RCNN_loss_bbox_ = fasterRCNN(im_data[:,6:,:,:,:],\
				# 												 im_info, \
				# 												 gt_boxes[:,6:,:,:],\
				# 												 num_boxes[:, 6:], input) # zl forward
				# track_rois = torch.cat([track_rois, track_rois_])
				# track_ids = torch.cat([track_ids,track_ids_])
				# track_cls = torch.cat([track_cls,track_cls_])
				# track_num = torch.cat([track_num,track_num_])	
				# track_cls_loss = torch.mean(torch.cat([track_cls_loss,track_cls_loss_]))
				# track_loc_loss = torch.mean(torch.cat([track_loc_loss,track_loc_loss_]))
				# track_predict_loc_loss = torch.mean(torch.cat([track_predict_loc_loss,track_predict_loc_loss_]))
				# rpn_loss_cls   = torch.mean(torch.cat([rpn_loss_cls,rpn_loss_cls_]))
				# rpn_loss_box   = torch.mean(torch.cat([rpn_loss_box,rpn_loss_box_]))
				# RCNN_loss_cls  = torch.mean(torch.cat([RCNN_loss_cls,RCNN_loss_cls_]))
				# RCNN_loss_bbox = torch.mean(torch.cat([RCNN_loss_bbox,RCNN_loss_bbox_]))	

			else:
				rois, cls_prob, bbox_pred, \
				rpn_loss_cls, rpn_loss_box, \
				RCNN_loss_cls, RCNN_loss_bbox,\
				rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes) # zl forward
				loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
					+ RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()	
						
				loss_temp += float(loss.item())
				# backward
				optimizer.zero_grad()
				loss.backward()
				if args.net == "vgg16":
						clip_gradient(fasterRCNN, 10.) # error prone
				optimizer.step()
			if step % args.disp_interval == 0:
				end = time.time()
				if step > 0:
					loss_temp /= (args.disp_interval +1 )
				if args.model != 'track':
					if args.mGPUs:
						loss_rpn_cls = rpn_loss_cls.mean().item()
						loss_rpn_box = rpn_loss_box.mean().item()
						loss_rcnn_cls = RCNN_loss_cls.mean().item()
						loss_rcnn_box = RCNN_loss_bbox.mean().item()
						fg_cnt = torch.sum(rois_label.data.ne(0))
						bg_cnt = rois_label.data.numel() - fg_cnt
					else:
						loss_rpn_cls = rpn_loss_cls.item()
						loss_rpn_box = rpn_loss_box.item()
						loss_rcnn_cls = RCNN_loss_cls.item()
						loss_rcnn_box = RCNN_loss_bbox.item()
						fg_cnt = torch.sum(rois_label.data.ne(0))
						bg_cnt = rois_label.data.numel() - fg_cnt

					print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
																	% (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
					print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
					print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
												% (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
					# call(["nvidia-smi"])
					# msg = "{}_disp_{:d}_finish".format(args.session, int(4))
					# output_alert(msg)
					sys.stdout.flush()
					if args.use_tfboard:
						info = {
							'loss': loss_temp,
							'loss_rpn_cls': loss_rpn_cls,
							'loss_rpn_box': loss_rpn_box,
							'loss_rcnn_cls': loss_rcnn_cls,
							'loss_rcnn_box': loss_rcnn_box
						}
						logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)
				else:
					if args.mGPUs:
						loss_track_cls = track_cls_loss.mean().item()
						loss_track_loc = track_loc_loss.mean().item()
						loss_track_predict = track_predict_loc_loss.mean().item()
						loss_rpn_cls = rpn_loss_cls.mean().item()
						loss_rpn_box = rpn_loss_box.mean().item()
						loss_rcnn_cls = RCNN_loss_cls.mean().item()
						loss_rcnn_box = RCNN_loss_bbox.mean().item()
					else:
						loss_track_cls = track_cls_loss.item()
						loss_track_loc = track_loc_loss.item()
						loss_track_predict = track_predict_loc_loss.item()
						loss_rpn_cls = rpn_loss_cls.item()
						loss_rpn_box = rpn_loss_box.item()
						loss_rcnn_cls = RCNN_loss_cls.item()
						loss_rcnn_box = RCNN_loss_bbox.item()

					print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
																	% (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
					print("\t\t\tnum_box:%d time cost: %f" % (num_boxes.float().view(-1).mean().item(), end-start))
					print("\t\t\ttrack_cls: %.4f, track_loc: %.4f, predict_loc: %.4f" \
												% (loss_track_cls,loss_track_loc,loss_track_predict))
					print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
												% (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
					call(["nvidia-smi"])
					sys.stdout.flush()
					if args.use_tfboard:
						info = {

							'loss': loss_temp,
							'loss_track_cls': loss_track_cls,
							'loss_track_loc': loss_track_loc,
							'loss_track_predict': loss_track_predict,
							'loss_rpn_cls': loss_rpn_cls,
							'loss_rpn_box': loss_rpn_box,
							'loss_rcnn_cls': loss_rcnn_cls,
							'loss_rcnn_box': loss_rcnn_box
						}
						logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)					
				# print("before del")
				# cpuStats()
				# gpuStats()
				# memReport()
				loss_temp = 0
				start = time.time()

		msg = "{}_{}_epoch_{:d}_finish".format(args.model, args.session, int(epoch))
		output_alert(msg)
		
		save_name = os.path.join(output_dir, '{}_{}_{}_{}.pth'.format(args.model, args.session, epoch, step)) 
		# zl save model after each batch
		save_checkpoint({
			'session': args.session,
			'epoch': epoch + 1,
			'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
			'optimizer': optimizer.state_dict(),
			'pooling_mode': cfg.POOLING_MODE,
			'class_agnostic': args.class_agnostic,
		}, save_name)
		print('save model: {}'.format(save_name))
		sys.stdout.flush()

	if args.use_tfboard:
		logger.close()


