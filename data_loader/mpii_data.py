import os
os.path.join("..")
import json
import random
import cv2 as cv
import mxnet as mx
import numpy as np

from tools.img_tool import *

class MPIIData(mx.gluon.data.Dataset):
    # 部分代码参考至 感谢提供了学习资源
    # https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras.git
    # https://github.com/princeton-vl/pose-hg-train.git 
    # https://github.com/bearpaw/pytorch-pose.git
    
    def __init__(self, jsonfile, imgpath, inres, outres, is_train, sigma=1, \
                    rot_flag=False, scale_flag=False, flip_flag=False):
        super(MPIIData, self).__init__()
        self.jsonfile = jsonfile    # json标注文件路径
        self.imgpath = imgpath      # 图片文件夹路径
        self.inres = inres
        self.outres = outres
        self.is_train = is_train
        self.nparts = 16
        self.anno = self._load_image_annotation()
        self.mean, self.std = self._get_mean_std()
        self.sigma = sigma
        self.rot_flag = rot_flag
        self.scale_flag = scale_flag
        self.flip_flag = flip_flag

    def _load_image_annotation(self):
        # load train or val annotation
        with open(self.jsonfile) as anno_file:
            anno = json.load(anno_file)

        val_anno, train_anno = [], []
        for idx, val in enumerate(anno):
            if val['isValidation'] == True:
                val_anno.append(anno[idx])
            else:
                train_anno.append(anno[idx])

        if self.is_train:
            return train_anno
        else:
            return val_anno

    def _get_mean_std(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        for item in self.anno:
            img_name = item["img_paths"]
            img_path = os.path.join(self.imgpath, img_name)
            img = cv.imread(img_path).transpose((2, 0, 1))   # HWC -> CHW
            mean += img.reshape(img.shape[2], -1).mean(axis=0)
            std += img.reshape(img.shape[2], -1).std(axis=1)
        return mean / len(self.anno), std / len(self.anno)

    def __getitem__(self, idx):

        if not self.is_train:
            assert (self.rot_flag == False), 'rot_flag must be off in val model'

        cropimg, gtmap, metainfo = self._process_img(idx)

        return cropimg, gtmap, metainfo

    def _process_img(self, idx):
        '''
        预处理图片
        '''
        img_name = self.anno[idx]["img_paths"]
        img_path = os.path.join(self.imgpath, img_name)
        img = cv.imread(img_path)   # HWC

        center = np.array(self.anno[idx]["objpos"])
        joints = np.array(self.anno[idx]["joint_self"])
        scale = self.anno[idx]['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        # 原理是???
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale *= 1.25

        # flip
        if self.flip_flag:
            img, joints, center = self.flip(img, joints, center)

        if self.scale_flag:
            scale = scale * np.random.uniform(0.8, 1.2)

        if self.rot_flag and random.choice([0, 1]):
            rot = np.random.randint(-1 * 30, 30)
        else:
            rot = 0

        cropimg = crop(img, center, scale, self.inres, rot)
        cropimg = normalize(cropimg, self.mean)

        # transform keypoints
        transformedKps = transform_kp(joints, center, scale, self.outres, rot)
        gtmap = generate_gtmap(transformedKps, self.sigma, self.outres)

        # meta info
        metainfo = {
            'sample_index': idx,
            'center': center,
            'scale': scale,
            'pts': joints, 
            'tpts': transformedKps, 
            'name': img_name
        }

        return cropimg, gtmap, metainfo

    def flip(self, img, joints, center):
        joints = np.copy(joints)
        matchedParts = [
            [0, 5],     # ankle
            [1, 4],     # knee
            [2, 3],     # hip
            [10, 15],   # wrise
            [11, 14],   # elbow
            [12, 13]    # shoulder
        ]

        _, width, _ = img.shape

        # flip image
        flip_img = cv.flip(img)

        # flip joints
        joints[:, 0] = width - joints[:, 0]

        for i, j in matchedParts:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

        # flip center
        flip_center = center
        flip_center[0] = width - center[0]

        return flip_img, joints, flip_center


    def __len__(self):
        return len(self.anno)