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

#SSDの推論時にconfとlocの出力から、被りを除去したBBoxを出力する
class Detect(Functional):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1) #confをソフトマックス関数で正規化するために用意
        self.conf_thresh = conf_thresh #confがconf_thresh=0.01より高いDBoxのみを扱う
        self.top_k = top_k #num_supressionでconfの高いtop_k個を計算に使用する
                           #top_k = 200
        self.nms_thresh = nms_thresh #nm_supressionでIOUがnms_thresh=0.45より大きいと同一物体へのBBoxとする

    def forward(self, loc_data, conf_data, dbox_list):
        """
        純伝搬の計算を実行する

        Parameters
        -----------
        loc_data: [batch_num, 8732, 4]
            オフセット情報
        conf_data: [batch_num, 8732, num_classes]
            検出の確信度
        dbox_list: [8732, 4]
            DBoxの情報

        Returns
        -----------
        output : torch.Size([batch_num, 21, 200, 5])
            (batch_num, class, confのtop200, BBoxの情報)
        """

        #各サイズを取得
        num_batch = loc_data.size(0) #ミニバッチのサイズ
        num_dbox = loc_data.size(1) #DBoxの数 = 8732
        num_classes = conf_data.size(2) #クラス数21

        #confはソフトマックスを適応して正規化する
        conf_data = self.softmax(conf_data)

        #出力の型を作成する。テンソルサイズは[minibatch数, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        #conf_dataを[batch_num, 8732, num_class]から[batch_num, num_classes, 8732]に順番を変更
        conf_preds = conf_data.transpose(2, 1)

        #ミニバッチごとのループ
        for i in range(num_batch):

            """### Step1 ###"""
            #locとDBoxから修正したBBox[xmin, ymin, xmax, ymax]を求める
            decode_boxes = decode(loc_data[i], dbox_list)

            #confのコピーを作成
            conf_scores = conf_preds[i].clone()
            """ ########### """
            
            #画像クラスごとのループ(背景クラスのindexである0は計算せず、index=1から)
            for cl in range(1, num_classes):

                """### Step2 ###"""
                #confの閾値を超えたBBoxを取り出す
                #confの閾値を超えているかをマスクを作成し、
                #閾値を超えたconfのインデックスをc_maskとして取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                #gtはGreater thanのこと。gtにより閾値を超えたものが１、超えないのが０になる
                #conf_scores:torch.Size([21, 8732])
                #c_mask:torch.Size([8732])

                #scoresはtorch.Size([閾値を超えたBBox数])
                scores = conf_scores[cl][c_mask]

                #閾値を超えたconfがない場合、つまりscores=[]のときは何もしない
                if scores.nelement() == 0: #nelementで要素数の合計を求める
                    continue

                #c_maskをdecoded_boxesに適応します
                boxes = decoded_boxes[l_mask].view(-1, 4)
                #decoded_boxes[l_mask]で1次元になってしまうので、
                #viewで(閾値を超えたBBox数, 4)サイズに変形し直す
                """ ############ """

                """### Step3 ###"""
                ids, count = nm_suppression(
                        boxes, scores, self.nms_thresh, self.top_k)
                #ids : confの降順にNon-Maximum Supressionを通過したBBoxの数

                #outputにNon-Maxmum Supressionを抜けた結果を格納
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[ids[:count]]), 1)
                """ ############ """

        return output #torch.Size([1, 21, 200, 5])
