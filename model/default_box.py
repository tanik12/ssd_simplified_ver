#よくわからんかったので後ほど。
import numpy as np
import pandas as pd
from itertools import product
from math import sqrt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import sys
sys.path.append('../')

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords,\
     PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

#デフォルトボックスを出力するクラス
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        #初期設定
        self.image_size = cfg['input_size'] #画像サイズの300
        #[38, 19, ...] 各sourceの特徴量マップのサイズ
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"]) #sourceの個数=6
        self.steps = cfg["steps"] #[8, 16, ...] DBoxのピクセルサイズ
        self.min_sizes = cfg["min_sizes"] #[30, 60, ...] 小さい正方形のDBoxのピクセルサイズ
        self.max_sizes = cfg["max_sizes"] #[60, 111, ...] 大きい正方形のDBoxのピクセルサイズ
        self.aspect_ratios = cfg["aspect_ratios"] #長方形のDBoxのアスペクト比

    def make_dbox_list(self):
        '''DBoxを作成する'''
        mean = []
        #'feature_maps':[38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2): #fまでの数で2ペアの組み合わせを作る

                #特徴量の画像サイズ
                #300 / 'step' : [8, 16, 32, 64, 100, 300]
                f_k = self.image_size / self.steps[k]

                #DBoxの中心座標x,y ただし、0~1で規格化している.
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #アスペクト比1の小さいDBox [cx, cy, width, height]
                #'min_sizes' : [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                #アスペクト比1の大きいDBox[cx, cy, width, height]
                #'max_sizes' : [45, 99, 153, 207, 261, 315]
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                #その他アスペクト比のdefBox[cx, cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        #DBoxをテンソルに変換。torch.size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        #DBoxが画像の外にはみ出るのを防ぐために、大きさを最小0, 最大1にする
        output.clamp_(max=1, min=0)

        return output

if __name__ == "__main__":
    #動作確認
    #SSD300の設定
    ssd_cfg = {
            'num_classes': 21, #背景クラスを含めた合計クラス数
            'input_size': 300, #画像の入力サイズ
            'bbox_aspect_num': [4, 6, 6, 6, 4, 4], #出力するDBoxのアスペクト比の種類
            'feature_maps': [38, 19, 10, 5, 3, 1], #各sourceの画像サイズ
            'steps': [8, 16, 32, 64, 100, 300], #DBOXの大きさを決める
            'min_sizes': [30, 60, 111, 162, 213, 264], #DBOXの大きさを決める
            'max_sizes': [60, 111, 162, 213, 264, 315], #DBOXの大きさを決める
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    # DBox作成
    dbox = DBox(ssd_cfg)
    dbox_list = dbox.make_dbox_list()
    
    #DBoxの出力を確認する
    print("DefaultBox :\n", pd.DataFrame(dbox_list.numpy()))
