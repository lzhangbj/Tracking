from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import time
import pdb
import PIL
import cPickle

from datasets.vid_eval import vid_eval
from datasets.imagenet_vid_eval_motion import vid_eval_motion
from .imagenet_vid_img_eval import vid_img_eval
import pickle
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class imagenet_vid_img(imdb):
    def __init__(self, image_set, devkit_path=None,data_path=None):
        imdb.__init__(self, "vid_img_"+image_set)
        print("imagenet_vid_img start")
        self._image_set = image_set  
        self._data_path = self._get_default_path() if data_path is None \
            else data_path
        self._devkit_path = devkit_path if devkit_path is not None else self._data_path+"/devkit"
        synsets_video = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_vid.mat'))
        self._vid=False

        self._classes = ('__background__',)
        self._wnid = (0,)
        for i in xrange(30):
            self._classes = self._classes + (synsets_video['synsets'][0][i][2][0],)
            self._wnid = self._wnid + (synsets_video['synsets'][0][i][1][0],)
            
        self._wnid_to_ind = dict(zip(self._wnid, xrange(31)))
        self._class_to_ind = dict(zip(self._classes, xrange(31)))

        self._image_ext = ['.JPEG']
        tic = time.clock()
        print("loading image index")
        sys.stdout.flush()
        self._image_index = self._load_image_set_index()
        tac = time.clock()
        print("loaded img index cost %ds" %(tac-tic))
        sys.stdout.flush()
        # Default to roidb handler
        tic = time.clock()
        print("loading roidb")
        sys.stdout.flush()
        self._roidb_handler = self.gt_roidb
        tac = time.clock()
        print("loaded roidb cost %ds" %(tac-tic))
        sys.stdout.flush()
        # self.frame_id=[]

        self._classes_map = ['__background__',  # always index 0
                        'n02691156', 'n02419796', 'n02131653', 'n02834778',
                        'n01503061', 'n02924116', 'n02958343', 'n02402425',
                        'n02084071', 'n02121808', 'n02503517', 'n02118333',
                        'n02510455', 'n02342885', 'n02374451', 'n02129165',
                        'n01674464', 'n02484322', 'n03790512', 'n02324045',
                        'n02509815', 'n02411705', 'n01726692', 'n02355227',
                        'n02129604', 'n04468005', 'n01662784', 'n04530566',
                        'n02062744', 'n02391049']

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000,
                       'matlab_eval': 1}

        assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def _get_default_path(self):
        return "/cluster/scratch/linzha/model/Data/ILSVRC2015"

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data','VID', self._image_set, index + self._image_ext[0])
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt

        if self._image_set == 'train':
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr_all.txt')
            image_index = []
            if os.path.exists(image_set_file):
                f = open(image_set_file, 'r')
                data = f.read().split()
                for lines in data:
                    if lines != '':
                        image_index.append(lines)
                f.close()
                return image_index #[:int(len(image_index)/100)]

            for i in range(1,31):
                image_set_file = os.path.join(self._data_path, 'ImageSets', 'VID', 'train_' + str(i) + '.txt')
                with open(image_set_file) as f:
                    tmp_index = [x.strip() for x in f.readlines()]
                    vtmp_index = []
                    for line in tmp_index:
                        line = line.split(' ')
                        image_list = os.popen('ls ' + self._data_path + '/Data/VID/train/' + line[0] + '/*.JPEG').read().split()
                        tmp_list = []
                        for imgs in image_list:
                            tmp_list.append(imgs[-63:-5])
                        vtmp_index = vtmp_index + tmp_list
                # num_lines = len(vtmp_index)
                # ids = np.random.permutation(num_lines)
                np.random.shuffle(vtmp_index)
                image_index+=vtmp_index
            np.random.shuffle(image_index)
            # for i in range(1,31):
            #     image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_pos_' + str(i) + '.txt')
            #     with open(image_set_file) as f:
            #         tmp_index = [x.strip() for x in f.readlines()]
            #     num_lines = len(tmp_index)
            #     ids = np.random.permutation(num_lines)
            #     count = 0
            #     while count < 2000:
            #         image_index.append(tmp_index[ids[count % num_lines]])
            #         count = count + 1
            image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr_all.txt')
            f = open(image_set_file, 'w')
            for lines in image_index:
                f.write(lines + '\n')
            f.close()
        else:
            image_set_file = os.path.join(self._data_path, 'ImageSets','VID', 'val.txt')
            with open(image_set_file) as f:
                lines = [x.strip().split(' ') for x in f.readlines()]
                image_index = [line[0] for line in lines]
                # i=0
                # for line in lines:
                #     if i>10:
                #         break
                #     print(i)
                #     print(line)
                #     sys.stdout.flush()
                #     i+=1
                self.frame_id = [int(line[1]) for line in lines]
                assert len(self.frame_id) == len(image_index), [len(self.frame_id) , len(image_index)]
                print(len(image_index))
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb_all.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb[:len(self._image_index)]
        print("loading gt_roidb from scratch")
        sys.stdout.flush()
        gt_roidb = []
        for i, index in enumerate(self.image_index):
            sys.stdout.write("\r %d / %d" % (i, len(self.image_index)))
            gt_roidb.append(self._load_imagenet_annotation(index))
            # gt_roidb = [self._load_imagenet_annotation(index)
            #         for indickle.HIGHEST_PROTOCOL)
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        sys.stdout.flush()
        assert len(gt_roidb) == self.num_images, [  len(gt_roidb) == self.num_images ]
        return gt_roidb

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._data_path, 'Annotations','VID', self._image_set, index + '.xml')
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        size = data.getElementsByTagName('size')[0]
        width = float(get_data_from_tag(size, 'width'))
        height = float(get_data_from_tag(size, 'height'))

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.int32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            x1 = float(get_data_from_tag(obj, 'xmin')) 
            y1 = float(get_data_from_tag(obj, 'ymin')) 
            x2 = float(get_data_from_tag(obj, 'xmax')) -1 
            y2 = float(get_data_from_tag(obj, 'ymax')) -1 
            assert x1>=0 and y1 >=0 and x2>=x1 and y2>=y1 and x2 != width and y2 != height, [x1, y1, x2, y2, width, height]
            cls = self._wnid_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'width':width,
                'height':height,
                'flipped' : False}

    # def _get_VID_results_file_template(self):
    #     # ILSVRC2015/results/<comp_id>_det_test_aeroplane.txt
    #     filename = '_vid_' + self._image_set + '_{:s}.txt'
    #     filedir = os.path.join(self._data_path, 'results')
    #     if not os.path.exists(filedir):
    #         os.makedirs(filedir)
    #     path = os.path.join(filedir, filename)
    #     return path

    # def _write_VID_results_file(self, all_boxes):
    #     # print(len(self.image_index))
    #     # print(len(self.frame_id))
    #     # for cls_ind, cls in enumerate(self.classes):
    #     #     if cls == '__background__':
    #     #         continue
    #     cls = 'all'
    #     print('Writing {} VID results file'.format(cls))
    #     filename = self._get_VID_results_file_template().format(cls)
    #     with open(filename, 'wt') as f:
    #         for cls_ind, cls in enumerate(self.classes):
    #             if cls == '__background__':
    #                 continue
    #             for im_ind, index in enumerate(self.image_index):
    #                 dets = all_boxes[cls_ind][im_ind]
    #                 if dets == []:
    #                     continue
    #                 # the VOCdevkit expects 1-based indices
    #                 for k in xrange(dets.shape[0]):
    #                     f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
    #                             format(self.frame_id[im_ind],\
    #                                     cls_ind, \
    #                                     dets[k, -1],\
    #                                    dets[k, 0], dets[k, 1],
    #                                    dets[k, 2], dets[k, 3]))

    # def _do_python_eval(self, output_dir='output'):
    #     info_str = ''
    #     annopath = os.path.join(
    #         self._data_path,
    #         'Annotations',
    #         'VID',
    #         '{:s}.xml')
    #     imagesetfile = os.path.join(
    #         self._data_path,
    #         'ImageSets',
    #         'VID',
    #         self._image_set + '.txt')
    #     cachedir = os.path.join(self._data_path, 'annotations_cache.pkl')
    #     aps = []
    #     if not os.path.isdir(output_dir):
    #         os.mkdir(output_dir)
    #     filename = self._get_VID_results_file_template().format('all')  
    #     aps = vid_img_eval(filename, annopath, imagesetfile, self._classes_map, cachedir, ovthresh=0.5)
    #     for i, cls in enumerate(self._classes):
            
    #         if cls == '__background__':
    #             continue
    #         ap = aps[i-1]
    #         print('AP for {} = {:.4f}'.format(cls, ap))
    #         with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
    #             cPickle.dump({'ap': ap}, f)
    #         info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
    #     print('Mean AP = {:.4f}'.format(np.mean(aps)))
    #     info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(aps))
    #     print('~~~~~~~~')
    #     print('Results:')
    #     for ap in aps:
    #         print('{:.3f}'.format(ap))
    #     print('{:.3f}'.format(np.mean(aps)))
    #     print('~~~~~~~~')
    #     print('')
    #     print('--------------------------------------------------------------')
    #     print('Results computed with the **unofficial** Python eval code.')
    #     print('Results should be very close to the official MATLAB eval code.')
    #     print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    #     print('-- Thanks, The Management')
    #     print('--------------------------------------------------------------')
    #     with open(os.path.join(output_dir, 'result'), 'wb') as f:
    #         f.write(info_str)

    # # def _do_matlab_eval(self, output_dir='output'):
    # #     print('-----------------------------------------------------')
    # #     print('Computing results with the official MATLAB eval code.')
    # #     print('-----------------------------------------------------')
    # #     path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
    # #                         'VOCdevkit-matlab-wrapper')
    # #     cmd = 'cd {} && '.format(path)
    # #     cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    # #     cmd += '-r "dbstop if error; '
    # #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
    # #         .format(self._devkit_path, self._get_comp_id(),
    # #                 self._image_set, output_dir)
    # #     print('Running:\n{}'.format(cmd))
    # #     status = subprocess.call(cmd, shell=True)

    # def evaluate_detections(self, all_boxes, output_dir):
    #     self._write_VID_results_file(all_boxes)
    #     self._do_python_eval(output_dir)
    #     # if self.config['matlab_eval']:
    #     #     self._do_matlab_eval(output_dir)
    #     if self.config['cleanup']:
    #         for cls in self._classes:
    #             if cls == '__background__':
    #                 continue
    #             filename = self._get_VID_results_file_template().format(cls)
    #             os.remove(filename)




    def _get_imagenetVid_results_file_template(self):
        # devkit/results/det_test_aeroplane.txt
        # filename = '_det_' + self._image_set + '_{:s}.txt'
        # base_path = os.path.join(self._devkit_path, 'results')
        # if not os.path.exists(base_path):
        #     os.mkdir(base_path)
        # path = os.path.join(
        #     self._devkit_path,
        #     'results',
        #     filename)
        # return path
        # ILSVRC2015/results/<comp_id>_det_test_aeroplane.txt
        filename = '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._data_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_imagenetVid_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} Imagenet vid results file'.format(cls))
            filename = self._get_imagenetVid_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._data_path,
            'Annotations','VID',self._image_set,
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets','VID',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_imagenetVid_results_file_template().format(cls)
            rec, prec, ap = vid_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
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

    def evaluate_detections(self, all_boxes, output_dir):
        
        #self._image_index = ['/'.join(roi_entry[0]['image'].split('/')[-3:])\
        #                        .replace('.JPEG','').replace('.jpeg', '')\
        #                        .replace('.jpg','').replace('.JPG','') \
        #                        for roi_entry in self._roidb]
        self._write_imagenetVid_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_imagenetVid_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True