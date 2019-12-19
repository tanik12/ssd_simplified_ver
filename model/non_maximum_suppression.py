#わからんかったまた見返す
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

#Non-Maximum Suppressionを行う関数
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maxinum Supressionを行う関数
    boxesのうち被り過ぎ(overlap以上)のBBoxを削除する

    Parameters
    ----------
    boxes : [確信度閾値 (0.01) を超えたBBox数, 4]
        BBox情報
    scores : [確信度閾値 (0.01) を超えたBBox数]
        confの情報

    Returns
    ----------
    keep : リスト
        confの降順にnmsを通過したindexが格納
    count : int
        nmsを通過したBBoxの数
    """

    #returnの雛形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    #keep : torch.Size([確信度閾値を超えたBBox数])、要素は全部０

    #各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    #boxesをコピーする。あとで、BBoxの被り度合いIOUの計算に使用する。
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    #scoreを昇順に並び替える
    v, idx = scores.sort(0)

    #上位top_k個(200個)のBBoxのindexを取り出す(200個存在しない場合もある)
    idx = idx[-top_k:]

    #idxの要素数が0でない限りループする
    while idx.numel() > 0:
        i = idx[-1] #現在のconf最大のindexをiに

        #keepの現在の最後にconf最大のindexを格納する
        #このindexのBBoxと被りが大きいBBoxをこれから消去する
        keep[count] = i
        count += 1

        #最後のBBoxになった場合は、ループを抜ける
        if idx.size(0) == 1;
            break

        #現在のconf最大のindexをkeepに格納したいので、idxを減らす
        idx = idx[:-1]

        # -------------------
        #これからkeepに格納したBBoxと被りの大きいBBoxを抽出して削除する
        #--------------------
        #1つ減らしたidxまでのBBoxをoutに指定した変数として作成する
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        #すべてのBBoxに対して、現在のBBox=indexがiと被っている値までに設定(clamp)
        #>>> a = torch.randn(4)
        #tensor([-1.7120,  0.1734, -0.0478, -0.0922])
        #>>> torch.clamp(a, min=-0.5, max=0.5)
        #tensor([-0.5000,  0.1734, -0.0478, -0.0922])
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        #wとhのテンソルサイズをindexを一つ減らしたものにする
        tmp_w.resize_as_(tmp_x2) #指定されたテンソルと同じサイズになるように変換
        tmp_h.resize_as_(tmp_y2) #指定されたテンソルと同じサイズになるように変換

        #clamした状態でのBBonの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        #clampされた状態での面積を求める
        inter = tmp_w * tmp_h
        
        #IoU = intersect部分 / (area(a) + area(b) - intersect部分)の計算
        rem_areas = torch.index_select(area, 0, idx) #各BBoxの元の面積
        union = (rem_areas - inter) + area[i] #2つのエリアのANDの面積
        IoU = inter/union

        #IoUがoverlapより小さいidxのみを残す
        idx = idx[IoU.le(overlap)] #leはLess than or Equal toの処理をする演算です
        #IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に
        #対してBBoxを囲んでいるために削除

    #whileのループを抜けたら終了
    return keep, count
