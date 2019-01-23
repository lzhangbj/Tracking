"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
import os
import sys
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb
import cPickle

def prepare_roidb(imdb, auto_load = True):
	"""Enrich the imdb's roidb by adding some derived quantities that
	are useful for training. This function precomputes the maximum
	overlap, taken over ground-truth boxes, between each ROI and
	each ground-truth box. The class with maximum overlap is also
	recorded.
	"""

	roidb = imdb.roidb
	# if not (imdb.name.startswith('coco')):
	# 	sizes = [PIL.Image.open(imdb.image_path_at(i)).size
	# 			 for i in range(imdb.num_images)]
	if not imdb._vid:
		path = os.path.join("prepared_final_img_roidb.pkl")
		if os.path.exists(path) and auto_load:
			with open(path, 'rb') as fid:
				imdb._roidb = cPickle.load(fid)
				return
		for i in range(imdb.num_images):
			roidb[i]['img_id'] = imdb.image_id_at(i)
			roidb[i]['image'] = imdb.image_path_at(i)
			# if not (imdb.name.startswith('coco')):
			# 	roidb[i]['width'] = sizes[i][0]
			# 	roidb[i]['height'] = sizes[i][1]
			# need gt_overlaps as a dense array for argmax
			gt_overlaps = roidb[i]['gt_overlaps'].toarray()
			# max overlap with gt over classes (columns)
			max_overlaps = gt_overlaps.max(axis=1)
			# gt class that had the max overlap
			max_classes = gt_overlaps.argmax(axis=1)
			roidb[i]['max_classes'] = max_classes
			roidb[i]['max_overlaps'] = max_overlaps
			# sanity checks
			# max overlap of 0 => class should be zero (background)
			zero_inds = np.where(max_overlaps == 0)[0]
			assert all(max_classes[zero_inds] == 0)
			# max overlap > 0 => class should not be zero (must be a fg class)
			nonzero_inds = np.where(max_overlaps > 0)[0]
			assert all(max_classes[nonzero_inds] != 0)
			sys.stdout.write("\r<<<<<<<<<<< prepare image loading {}/{} >>>>>>>>>>".format(i+1, imdb.num_images))
			sys.stdout.flush()
		sys.stdout.write("\n")
		imdb._roidb = roidb
		with open(path, 'wb') as fid:
			cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
	else:
		path = os.path.join("prepared_final_vid_roidb_{}_quater.pkl".format(imdb._seq_length))
		if os.path.exists(path) and auto_load:
			with open(path, 'rb') as fid:
				imdb._roidb = cPickle.load(fid)
				return
		for video_index in range(len(imdb._image_index)):
			for frame_index in range(len(imdb._image_index[video_index])):
				roidb[video_index][frame_index]['img_id'] = imdb.image_id_at(video_index, frame_index)
				roidb[video_index][frame_index]['image'] = imdb.image_path_at(video_index, frame_index)
				# if not (imdb.name.startswith('coco')):
				# 	roidb[i]['width'] = sizes[i][0]
				# 	roidb[i]['height'] = sizes[i][1]
				# need gt_overlaps as a dense array for argmax
				gt_overlaps = roidb[video_index][frame_index]['gt_overlaps'].toarray()
				# max overlap with gt over classes (columns)
				max_overlaps = gt_overlaps.max(axis=1)
				# gt class that had the max overlap
				max_classes = gt_overlaps.argmax(axis=1)
				roidb[video_index][frame_index]['max_classes'] = max_classes
				roidb[video_index][frame_index]['max_overlaps'] = max_overlaps
				# sanity checks
				# max overlap of 0 => class should be zero (background)
				zero_inds = np.where(max_overlaps == 0)[0]
				assert all(max_classes[zero_inds] == 0)
				# max overlap > 0 => class should not be zero (must be a fg class)
				nonzero_inds = np.where(max_overlaps > 0)[0]
				assert all(max_classes[nonzero_inds] != 0)
				sys.stdout.write("\r<<<<<<<<<<< prepare image loading {}/{} >>>>>>>>>>".format(video_index+1, len(imdb._image_index)))
				sys.stdout.flush()
		sys.stdout.write("\n")
		imdb._roidb = roidb
		with open(path, 'wb') as fid:
			cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)		


def rank_roidb_ratio(roidb, imdb):
		# rank roidb based on the ratio between width and height.
		ratio_large = 2 # largest ratio to preserve.
		ratio_small = 0.5 # smallest ratio to preserve.    
		
		ratio_list = []
		if imdb._vid:
			for oo, i in enumerate(imdb._video_structure):
				width = roidb[oo][0]['width']
				height = roidb[oo][0]['height']
				ratio = width / float(height)
				if ratio > ratio_large:
					roidb[oo][0]['need_crop'] = 1
					ratio = ratio_large
				elif ratio < ratio_small:
					roidb[oo][0]['need_crop'] = 1
					ratio = ratio_small        
				else:
					roidb[oo][0]['need_crop'] = 0
				ratio_list.append(ratio)
				# from second frame onwards, make sure the images in the same video have same shape
				for j in range(1, i):
					assert roidb[oo][j]['width'] == width, "image's width in the same video should be the same"
					assert roidb[oo][j]['height'] == height, "image's height in the same video should be the same"
				sys.stdout.write("\rranking {:d}/{:d}".format(oo+1, len(imdb._video_structure)))
				sys.stdout.flush()
			print()


		else:
			for i in range(len(roidb)):
				width = roidb[i]['width']
				height = roidb[i]['height']
				ratio = width / float(height)

				if ratio > ratio_large:
					roidb[i]['need_crop'] = 1
					ratio = ratio_large
				elif ratio < ratio_small:
					roidb[i]['need_crop'] = 1
					ratio = ratio_small        
				else:
					roidb[i]['need_crop'] = 0

				ratio_list.append(ratio)

		ratio_list = np.array(ratio_list)
		ratio_index = np.argsort(ratio_list)
		return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb, imdb):
		# filter the image/videos without bounding box.
		print('before filtering, there are %d images...' % (len(roidb)))
		i = 0
		while i < len(roidb):
			if not imdb._vid:
				if len(roidb[i]['boxes']) == 0:
					del roidb[i]
					del imdb._image_index[i]
					i -= 1
				i += 1
			else:
				assert len(roidb)==len(imdb._video_structure), [len(roidb), len(imdb._video_structure)]
				for j in range(imdb._seq_length):
					if len(roidb[i][j]['boxes']) == 0:
						del roidb[i]
						del imdb._video_structure[i]
						del imdb._image_index[i]
						# del imdb._widths[i]
						i -= 1
						break
				i += 1				
			sys.stdout.write("\rfiltering {:d}/{:d}".format(i+1, len(roidb)))
			sys.stdout.flush()
		print()
		print('after filtering, there are %d images/videos...' % (len(roidb)))
		sys.stdout.flush()
		return roidb

def combined_roidb(imdb_names, training=True):
	"""
	Combine multiple roidbs
	"""

	def get_training_roidb(imdb):
		"""Returns a roidb (Region of Interest database) for use in training."""
		if cfg.TRAIN.USE_FLIPPED:
			print('Appending horizontally-flipped training examples...')
			imdb.append_flipped_images()
			print('done')

		# print('Preparing training data...')
		# sys.stdout.flush()
		prepare_roidb(imdb)
		#ratio_index = rank_roidb_ratio(imdb)
		print('done')
		return imdb.roidb
	
	def get_roidb(imdb_name):
		imdb = get_imdb(imdb_name)
		print('Loaded dataset `{:s}` for training'.format(imdb.name))
		sys.stdout.flush()
		imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
		print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
		sys.stdout.flush()
		roidb = get_training_roidb(imdb)
		sys.stdout.flush()
		return imdb, roidb

	imdbs_roidbs = [get_roidb(s) for s in imdb_names.split('+')]
	roidbs = [t[1] for t in imdbs_roidbs]
	imdbs = [t[0] for t in imdbs_roidbs]
	roidb = roidbs[0]
	
	is_vid = np.sum([int(t._vid) for t in imdbs])
	if not is_vid:
		if len(roidbs) > 1:
			for r in roidbs[1:]:
				roidb.extend(r)
			tmp = get_imdb(imdb_names.split('+')[1])
			imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
		else:
			imdb = imdbs[0]
	else:
		imdb = imdbs[0]

	if training:
		roidb = filter_roidb(roidb, imdb)

	ratio_list, ratio_index = rank_roidb_ratio(roidb, imdb)


	# if training:
	# 	if not is_vid:
	# 		roidb = filter_roidb(roidb, imdb)
	# 		if len(roidbs) > 1:
	# 			for r in roidbs[1:]:
	# 				roidb.extend(r)
	# 			tmp = get_imdb(imdb_names.split('+')[1])
	# 			imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
	# 		else:
	# 			imdb = get_imdb(imdb_names)
	# 	else:
	# 		imdb=imdbs[0]
	# 		roidb = roidbs[0]
	# 		roidb = filter_roidb(roidb, imdb)
	# 		if os.path.exists("data/cache/roidb_vid2015_store.npy") and False:
	# 			roidb = np.load("data/cache/roidb_vid2015_store.npy")
	# 			ratio_list = np.load("data/cache/ratio_list_vid2015_store.npy")
	# 			ratio_index= np.load("data/cache/ratio_index_vid2015_store.npy")
	# 		else:
	# 			ratio_list, ratio_index = rank_roidb_ratio(roidb, imdb)
	# 			np.save("data/cache/roidb_vid2015_store.npy",roidb)
	# 			np.save("data/cache/ratio_list_vid2015_store.npy",ratio_list)
	# 			np.save("data/cache/ratio_index_vid2015_store.npy",ratio_index)


	return imdb, roidb, ratio_list, ratio_index
