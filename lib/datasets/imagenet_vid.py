from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom
import time
import os
import sys
import cPickle
import PIL
import cv2
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval
from .imagenet_vid_eval import vid_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



class imagenet_vid(imdb):
    def __init__(self, image_set, seq_length=6, img_index_auto_load=True,gt_roidb_auto_load=True, data_path=None):
        """
        fill basic information to initialize imdb
        """
        imdb.__init__(self, 'imagenet_vid_2015')
        self._image_set = image_set     # imageset = 'train' or 'test'
        self._gt_roidb_auto_load=gt_roidb_auto_load
        self._img_index_auto_load = img_index_auto_load
        self._data_path = self._get_default_path() if data_path is None \
            else data_path
        # self._widths = []
        self._classes = ['__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra']
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._video_structure = []
        self._vid = True
        self._image_index = []
        self._load_image_set_index()
        self._seq_length = seq_length
        self._gt_track_ids = []
        
        # if seq_length is not None:
        #     self._image_index = self._load_lstm_image_index(seq_length)
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': False,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,   #################################  waiting to be filled Forrest
                       'min_size': 2}

        assert os.path.exists(self._data_path), \
            'Data Path does not exist: {}'.format(self._data_path)


        self._classes_map = ['__background__',  # always index 0
                        'n02691156', 'n02419796', 'n02131653', 'n02834778',
                        'n01503061', 'n02924116', 'n02958343', 'n02402425',
                        'n02084071', 'n02121808', 'n02503517', 'n02118333',
                        'n02510455', 'n02342885', 'n02374451', 'n02129165',
                        'n01674464', 'n02484322', 'n03790512', 'n02324045',
                        'n02509815', 'n02411705', 'n01726692', 'n02355227',
                        'n02129604', 'n04468005', 'n01662784', 'n04530566',
                        'n02062744', 'n02391049']

    def img_index_to_vid_index(self, i):
        assert i < self.num_images, "img index out of range"
        order = 0
        while i-self._video_structure[order] >= 0:
            i-=self._video_structure[order]
            order+=1
        return order

    def _get_widths(self):
        sys.stdout.write("<<<<<<<<<<< get image width loading {}/{} >>>>>>>>>>".format(0, len(self._image_index)))
        img_count=0
        hh=[]
        for i in range(len(self._image_index)):
            width = PIL.Image.open(self.image_path_at(img_count)).size[0]
            hh.append(width)
            img_count+=len(self._image_index[i])
            sys.stdout.write("\r<<<<<<<<<<< get image width loading {}/{} >>>>>>>>>>".format(i+1, len(self._image_index)))
            sys.stdout.flush()
        sys.stdout.write("\r")
        return hh

    def append_flipped_images(self):
        tic = time.clock()
        assert self.roidb is not None
        num_images = self.num_images
        # widths = self._widths
        for v in range(len(self._video_structure)):
            new_roidb = []
            for f in range(self._video_structure[v]):
                assert f<self._seq_length, f
                boxes = self.roidb[v][f]['boxes'].copy()
                widths = self.roidb[v][f]['width']
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = widths - oldx2 -1
                boxes[:, 2] = widths - oldx1 -1
                # assert self.roidb[i]['image'] == self.image_path_at(i), [ self.roidb[i]['image'], self.image_path_at(i) ]
                # assert widths[v] == self.roidb[v][f]['width'],[ widths[v], self.roidb[v][f]['width'], v,f ]
                assert (boxes[:, 2] >= boxes[:, 0]).all(), [ self.image_path_at(v,f), \
                                                            self.roidb[v][f]['boxes'][:, 0], self.roidb[v][f]['boxes'][:,2], \
                                                            boxes[:, 2], boxes[:, 0] ]
                entry = {\
                # 'image': self.roidb[v][f]['image'],
                # 'img_id':self.roidb[v][f]['img_id'],
                'width': self.roidb[v][f]['width'],
                'height':self.roidb[v][f]['height'],
                'boxes': boxes,
                'gt_overlaps': self.roidb[v][f]['gt_overlaps'],
                'gt_classes': self.roidb[v][f]['gt_classes'],
                # 'max_overlaps': self.roidb[v][f]['max_overlaps'],
                # 'max_classes': self.roidb[v][f]['max_classes'],
                'flipped': True}           
                new_roidb.append(entry)
            self.roidb.append(new_roidb)
            sys.stdout.write("\r<<<<<<<<<<< flip image loading {}/{} >>>>>>>>>>".format(v, len(self._video_structure)))
            sys.stdout.flush()
        print()
        self._image_index = self._image_index * 2
        self._video_structure*=2
        toc = time.clock()
        print("flipped image cost {:.1f}s".format(toc-tic))
        sys.stdout.flush()

    def image_index_to_video_index(self, i):
        assert i < self.num_images, "index out of images length range"
        for order,j in enumerate(self._video_structure):
            if i-j<0:
                return order, i
            else:
                i-=j
        print("image index wrong!")
        raise       
   
    def video_index_to_image_index(self, v, f):
        return np.sum(self._video_structure[:v]) + f
  
    def image_path_at(self, i, f=None):
        if f is not None:
           return self.image_path_from_index(self._image_index[i][f])
        assert i < self.num_images, "index out of images length range"
        v, f = self.image_index_to_video_index(i)
        return self.image_path_at(v,f)

    def image_id_at(self, i,f=None):
        """
        Return the absolute path to image i in the image sequence.
        """
        if f is None:
            return i
        return self.image_id_at(self.video_index_to_image_index(i,f))

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data','VID',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        Example path to image set file:
        data/ILSVRC2015/ImageSets/VID/train.txt
        style of txt file:
        video_set_location    frame
        video_set_location    frame
        end
        video_set_location    frame
        ......
        where end means the end of one video clip
        
        return image_index[video_index][frame_index] = path to frame file without extension
        """

        image_index_path = os.path.join(self.cache_path , "ILSVRC2015_image_index.pkl")
        # widths_path = os.path.join(self.cache_path , "ILSVRC2015_image_widths.pkl")
        ################# load saved data ###################
        if os.path.exists(image_index_path) and self._img_index_auto_load:
            print("loading image index from cache")
            sys.stdout.flush()
            self._image_index = cPickle.load(open(image_index_path,"rb"))
            ############### load partial example ################# ##############################
            # np.random.shuffle(self._image_index)
            self._image_index = self._image_index[:int(len(self._image_index)/4)]
            self._video_structure[:] = []
            for v in self.image_index:
                self._video_structure.append(len(v))
            # self._widths = cPickle.load(open(widths_path,"rb"))
            # self._widths=self._get_widths()
            # cPickle.dump(self._widths, open(widths_path, "wb"))
            # print('wrote image width to {}'.format(widths_path))
            return 
        ################# create and save data ###################
        print("loading image index from raw")
        sys.stdout.flush()
        image_index = []
        self._video_structure=[]
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'VID',
                                      self._image_set + '.txt')
        image_data_dir = os.path.join(self._data_path, 'Data', 'VID')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            video_index = []
            for line in f.readlines():
                line = os.path.join(self._image_set, line.strip().split()[0])
                loc = os.path.join(image_data_dir, line)
                video_index = []
                i = 0
                while os.path.exists(os.path.join(loc, "{:0>6d}.JPEG".format(i))):
                    video_index.append(os.path.join(line, "{:0>6d}".format(i)))
                    i+=1
                self._video_structure.append(len(video_index))
                image_index.append(video_index[:])
        ### save image_index ###
        cPickle.dump(image_index, open(image_index_path, "wb"))
        print('wrote image_index to {}'.format(image_index_path))
        self._image_index = image_index
        # self._widths=self._get_widths()
        # cPickle.dump(self._widths, open(widths_path, "wb"))
        # print('wrote image width to {}'.format(widths_path))
        sys.stdout.flush()


    def _get_default_path(self): #
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return "/cluster/scratch/linzha/model/Data/ILSVRC2015"

    @property
    def num_images(self):
        return sum(self._video_structure)

    def _load_vid_annotation(self, video_index, frame_index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record
        ['image', 'height', 'width', 'boxes', 'gt_classes', 'gt_overlaps', 'flipped']

        image       : image_index e.g. "train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00030000/000000"
        boxes       : (num_valid_objs, 4) the 4 is (x1, y1, x2, y2)
        gt_classes  : (num_valid_objs,) is class index
        gt_overlaps : (num_valid_objs, num_classes) non_exist classes is 0, exist is 1.0
        flipped     : False

        we assume loading raw video first then transfers to lstm structure
        """
        index = self._image_index[video_index][frame_index]
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        # roi_rec['image'] = self.image_path_from_index(index) # the image directory, including .jpg

        filename = os.path.join(self._data_path, 'Annotations', 'VID', index + '.xml')

        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        # assert roi_rec['width']==self._widths[video_index]
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 5), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        valid_objs = np.zeros((num_objs), dtype=np.bool)

        class_to_index = dict(zip(self._classes_map, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = np.maximum(float(bbox.find('xmin').text), 0) 
            y1 = np.maximum(float(bbox.find('ymin').text), 0) 
            x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width']-1) -1
            y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1) -1
            if not class_to_index.has_key(obj.find('name').text):
                continue
            valid_objs[ix] = True
            cls = class_to_index[obj.find('name').text.lower().strip()]
            trackid=int(obj.find('trackid').text)
            boxes[ix, :] = [x1, y1, x2, y2, trackid]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        boxes = boxes[valid_objs, :]
        gt_classes = gt_classes[valid_objs]
        overlaps = overlaps[valid_objs, :]
        num_valid_boxes = np.sum(valid_objs)
        overlaps = scipy.sparse.csr_matrix(overlaps)

        assert (boxes[:, 2] >= boxes[:, 0]).all()

        roi_rec.update({'boxes': boxes, 
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'flipped': False,
                        'seg_areas': np.zeros((num_valid_boxes,), dtype=np.float32)}) # no seg label in vid dataset.
        return roi_rec

    def gt_roidb(self, transfer = True):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        

        """
        tic = time.clock()
        cache_file = os.path.join(self.cache_path, self.name + '_{}_gt_roidb.pkl'.format(3))
        if os.path.exists(cache_file) and self._gt_roidb_auto_load:
            with open(cache_file, 'rb') as fid:
                gt_roidb = cPickle.load(fid)
            gt_roidb = gt_roidb[:len(self._image_index)]
            toc = time.clock()
            print('{} gt roidb loaded from {} cost {:.1f}s'.format(self.name, cache_file, toc-tic))
            sys.stdout.flush()
            # max_track_id=0
            # for gt in gt_roidb:
            #     for v_gt in gt:
            #         if len(v_gt['boxes']) != 0:
            #             max_id = np.amax(v_gt['boxes'][:, 4])
            #             max_track_id = max(max_track_id, max_id)
            # print("max track id")
            # print(max_track_id)
        else:
            gt_roidb = []
            sys.stdout.write("<<<<<<<<<<< get image gt roidb loading {}/{} >>>>>>>>>>".format(0, len(self._image_index)))
            for video_index in range(len(self._image_index)):
                video_roidbs = []
                # assert len(self._image_index[video_index])==self._seq_length
                for frame_index in range(len(self._image_index[video_index])):
                    roidb = self._load_vid_annotation(video_index, frame_index)
                    ### prepare roidb include more features ###
                    # roidb['img_id'] = self.image_id_at(video_index, frame_index)
                    # gt_overlaps = roidb['gt_overlaps'].toarray()
                    # max overlap with gt over classes (columns)
                    # max_overlaps = gt_overlaps.max(axis=1)
                    # gt class that had the max overlap
                    # max_classes = gt_overlaps.argmax(axis=1)
                    # roidb['max_classes'] = max_classes
                    # roidb['max_overlaps'] = max_overlaps
                    # sanity checks
                    # max overlap of 0 => class should be zero (background)
                    # zero_inds = np.where(max_overlaps == 0)[0]
                    # assert all(max_classes[zero_inds] == 0)
                    # max overlap > 0 => class should not be zero (must be a fg class)
                    # nonzero_inds = np.where(max_overlaps > 0)[0]
                    # assert all(max_classes[nonzero_inds] != 0)
                    video_roidbs.append(roidb)
                    # assert video_roidbs[-1]['image'] == self.image_path_at(video_index, frame_index),[video_index, frame_index]
                    assert video_roidbs[0]['boxes'].shape[1] == 5, video_roidbs[0]['boxes'].shape[1]
                gt_roidb.append(video_roidbs)
                sys.stdout.write("\r<<<<<<<<<<< get image gt roidb loading {}/{} >>>>>>>>>>".format(video_index+1, len(self._image_index)))
                sys.stdout.flush()

            with open(cache_file, 'wb') as fid:
                cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
            toc = time.clock()
            print("time cost {:.1f}s".format(toc-tic))
            print('wrote gt roidb to {}'.format(cache_file))
            sys.stdout.flush()
        ################# roidb to seq length if necessary ##################33
        if self._seq_length is not None and transfer:
            seq_length=self._seq_length
            image_index = []
            self._video_structure[:] = []
            print("transfering lstm seq {} video structure".format(self._seq_length))
            tic = time.clock()
            sys.stdout.flush()
            new_hh = []
            new_gt_roidb = []
            assert len(self._image_index) != 0
            for tt, v in enumerate(self._image_index):
                if len(v) < self._seq_length:
                    continue
                i = 0
                while i+seq_length-1 < len(v):
                    # new_hh.append(self._widths[tt])
                    image_index.append(v[i:i+seq_length])
                    new_gt_roidb.append(gt_roidb[tt][i:i+seq_length])
                    assert len(image_index[-1]) == self._seq_length, self._image_path_at(tt, i)
                    i+=seq_length

                if i!= len(v):
                    # new_hh.append(self._widths[tt])
                    image_index.append(v[len(v)-seq_length:len(v)])
                    new_gt_roidb.append(gt_roidb[tt][len(v)-seq_length:len(v)])
                    assert len(image_index[-1]) == self._seq_length, self._image_path_at(tt, i)

            self._video_structure = [seq_length]*len(image_index)
            # self._widths=new_hh
            self._image_index=image_index
            gt_roidb=new_gt_roidb
            # cPickle.dump(self._widths, open(widths_path, "wb"))
            # print('wrote image width to {}'.format(widths_path))
            toc = time.clock()
            print("transfering finished, cost {:.1f}s".format(toc-tic))
            sys.stdout.flush()
        assert gt_roidb[0][0]['boxes'].shape[1] == 5, gt_roidb[0][0]['boxes'].shape[1]
        return gt_roidb

    def rpn_roidb(self):              
        '''
            load pretrained rpn roidb from rpn file
            in training mode we mix gt boxes and rpn boxes to accelarate training  
            waiting to befilled
        '''               
        # if int(self._year) == 2007 or self._image_set != 'test':
        tic=time.clock()
        sys.stdout.write("loading rpn roidb\n")
        sys.stdout.flush()
        if self._image_set != 'test':                 
            gt_roidb = self.gt_roidb(transfer=False)
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
            if self._seq_length is not None:
                seq_length=self._seq_length
                image_index = []
                self._video_structure[:] = []
                print("transfering lstm seq {} video structure".format(self._seq_length))
                sys.stdout.flush()
                new_hh = []
                new_roidb = []
                assert len(self._image_index) != 0
                for tt, v in enumerate(self._image_index):
                    if len(v) < self._seq_length:
                        continue
                    i = 0
                    while i+seq_length-1 < len(v):
                        # new_hh.append(self._widths[tt])
                        image_index.append(v[i:i+seq_length])
                        new_roidb.append(roidb[tt][i:i+seq_length])
                        assert len(image_index[-1]) == self._seq_length, self._image_path_at(tt, i)
                        i+=seq_length
                    if i!= len(v):
                        # new_hh.append(self._widths[tt])
                        image_index.append(v[len(v)-seq_length:len(v)])
                        new_roidb.append(roidb[tt][len(v)-seq_length:len(v)])
                        assert len(image_index[-1]) == self._seq_length, self._image_path_at(tt, i)
                self._video_structure = [seq_length]*len(image_index)
                ### save image_index ###
                # cPickle.dump(image_index, open(lstm_image_index_path, "wb"))
                # print('wrote lstm image_index to {}'.format(lstm_image_index_path)) i think its quick enough, no need to speed up again
                # sys.stdout.flush()
                # self._widths=new_hh
                self._image_index=image_index
                roidb=new_roidb
                print("transfering finished")
                sys.stdout.flush()
        else: # no need to transfer in testing mode
            roidb = self._load_rpn_roidb(None)
        toc = time.clock()
        sys.stdout.write("loaded rpn roidb time {:.1f}s\n".format(toc-tic))
        sys.stdout.flush()
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        sys.stdout.flush()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_VID_results_file_template(self):
        # ILSVRC2015/results/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._data_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_VID_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VID results file'.format(cls))
            filename = self._get_VID_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        info_str = ''
        annopath = os.path.join(
            self._data_path,
            'Annotations',
            'VID',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            'VID',
            self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache.pkl')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        filename = self._get_VID_results_file_template().format('all')  
        aps, rec, prec = vid_eval(filename, annopath, imagesetfile, self._classes_map, cachedir, ovthresh=0.5)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('AP for {} = {:.4f}'.format(cls, aps[i-1]))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': aps}, f)
            info_str += 'AP for {} = {:.4f}\n'.format(cls, aps[i-1])
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(aps))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')
        with open(os.path.join(output_dir, 'result'), 'wb') as f:
            f.write(info_str)

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_VID_results_file(all_boxes)
        self._do_python_eval(output_dir)
        # if self.config['matlab_eval']:
        #     self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_VID_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

###################################################################################################
 