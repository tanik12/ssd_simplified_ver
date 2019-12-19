import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from get_datapath import make_datapath_list
import cv2
import xml.etree.ElementTree as ET 

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#XML形式のアノテーションをリスト形式に変換するクラス

class Anno_xml2list(object):
    """
    1枚の画像に対するXML形式のアノテーションデータを、画像サイズで規格化してからlist形式にする。

    Attributes
    ----------
    classes : リスト
        VOCのクラス名を格納したリスト
    """

    def __init__(self, classes):
        self.classes = classes
    def __call__(self, xml_path, width, height):
        """
        一枚の画像に対するXML形式のアノテーションデータを画像サイズで規格化してからリスト形式に変換する。

        Parameters
        ----------
        xml_path : str
            cmlファイルへのパス。
        width : int
            対象画像の幅。
        height : int
            対象画像の高さ。

        Returns
        -------
        ret : [[xmin, ymin, ymax, xmax, label_ind], ... ]
            物体のアノテーションデータを格納したリスト。画像内に存在する物体数分だけ要素を持つ。
        """

        #画像内のずべての物体のアノテーションをこのリストに格納します
        ret = []

        #xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        #画像内にある物体(object)の数だけループする
        for obj in xml.iter('object'):
            #アノテーションで検知がdifficultんい雪堤されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            
            #1つの物体に対するアノテーションを格納するリスト
            bndbox = []

            name = obj.find('name').text.lower().strip() #物体名
            bbox = obj.find('bndbox') #バウンディングボックスの情報

            #アノテーションのxmin, ymin, xmax, ymaxを取得し、0~1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                #VOCは原点が(1, 1)なので1を引き算して(0, 0)に
                cur_pixel = int(bbox.find(pt).text) - 1

                #幅、高さで規格化
                if pt == 'xmin' or pt == 'xmax': #x方向の時は幅で割り算
                    cur_pixel /= width
                else:
                    cur_pixel /= height #y方向の時は高さで割り算

                bndbox.append(cur_pixel)

            #アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            #resに[xmin, ymin, xmax, ymax, label_ind]を足す
            ret += [bndbox]

        return np.array(ret) #[[xmin, ymin, ymax, xmax, label_ind], ... ]

if __name__ == "__main__":
    rootpath = "/Users/gisen/git/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
    
    #動作確認
    voc_classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    transform_anno = Anno_xml2list(voc_classes)

    #画像の読み込み OpemCVを使用
    ind = 1
    image_file_path = train_img_list[ind]
    img = cv2.imread(image_file_path) #[高さ][幅][色RGB]
    height, width, channels = img.shape #画像のサイズを取得

    #アノテーションをリストで表示
    print(transform_anno(val_anno_list[ind], width, height))
