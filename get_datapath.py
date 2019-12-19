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

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

def make_datapath_list(rootpath):
    """
    データへのパスを格納したリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    ----------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """

    #画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    #訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    #訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip() #空白スペースと改行を除去
        img_path = (imgpath_template % file_id) #画像のパス
        anno_path = (annopath_template % file_id) #アノテーションのパス
        train_img_list.append(img_path) #listに追加
        train_anno_list.append(anno_path) #listに追加

    #検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip() #空白スペースと改行を除去
        img_path = (imgpath_template % file_id) #画像のパス
        anno_path = (annopath_template % file_id) #アノテーションのパス
        val_img_list.append(img_path) #リストに追加
        val_anno_list.append(anno_path) #リストに追加

    return train_img_list, train_anno_list, val_img_list, val_anno_list

if __name__ == '__main__':
    rootpath = "/Users/gisen/git/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
    print(train_img_list[0])
