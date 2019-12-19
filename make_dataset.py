import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords,\
PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

from get_anno_data import Anno_xml2list
from get_datapath import make_datapath_list
from processing import DataTransform 
import cv2
import xml.etree.ElementTree as ET

#VOC2012のDatasetを作成する

class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。Pytorchのクラスを継承。
    
    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    transform_anno : object
        xmlのアノテーションをリストに変換するインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase #train or val
        self.transform = transform #画像の変形
        self.transform_anno = transform_anno #アノテーションデータをxmlからリストへ

    def __len__(self):
        '''画像の枚数を返す '''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理した画像のテンソル形式のテータ、アノテーションを取得する。
        '''
        img, gt, h, w = self.pull_item(index)
        return img, gt

    def pull_item(self, index):
        '''
        前処理した画像のテンソル形式のテータ、アノテーション、画像の高さ、幅を取得する。
        '''

        #画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path) #[高さ][幅][chanel]
        height, width, channels = img.shape #画像のサイズ

        #xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        #前処理を実施
        img, boxes, labels = self.transform(
                img, self.phase, anno_list[:, :4], anno_list[:, 4])
        #色チャネルの順番がBGRになっているので,RGBに順番を変更
        #さらに(高さ、幅、色チャネル)の順を(色チャネル、高さ、幅)に変換
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        #BBoxとラベルをセットにしたnp.arrayを作成、変数名gtはground truth(答え)の略称
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

if __name__ == "__main__":
    #動作確認
    #画像の読み込み
    rootpath = "/Users/gisen/git/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    #アノテーションをリストに
    voc_classes = ['aeroplane','bicycle','bird','boat','bottle','bus',\
                   'car','cat','chair','cow','diningtable','dog','horse',\
                   'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    color_mean = (104, 117, 123) #(BGR)の色の平均値
    input_size = 300

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",\
                               transform=DataTransform(input_size, color_mean),\
                               transform_anno=Anno_xml2list(voc_classes))

    val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',\
                             transform=DataTransform(input_size, color_mean),\
                             transform_anno=Anno_xml2list(voc_classes))

    #データの取り出し例
    print(val_dataset.__getitem__(1))
