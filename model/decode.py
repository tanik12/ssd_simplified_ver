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

#オフセット情報を使い、DBoxに変換する関数
def decode(loc, dbox_list):
    """
    オフセット情報を使い、DBoxをBBoxに変換する。

    Parameters
    ----------
    loc: [8732, 4]
        SSDモデルで推論するオフセット情報。
    dbox_list: [8732, 4]
        DBoxの情報
    
    Returns
    ----------
    boxes : [xmin, ymin, xmax, ymax]
        BBoxの情報
    """

    #DBoxは[cx, cy, width, height]で格納されている
    #locも[Δcx, Δcy, Δwidth, Δheight]で格納されている
    #BBoxの情報は下記により計算(本は間違っている
    #cx = cx_d + 0.1 Δcx * w_d
    #cy = cy_d + 0.1 Δcy * h_d
    #w = w_d * exp (0.2 Δw)
    #h = h_d * exp (0.2 Δh)

    #オフセット情報からBBoxを求める
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:,2:] * 0.2)), dim=1)
    #boxesのサイズはtorch.Size([8732, 4])となります。

    #BBoxの座標情報を[cx, cy, width, height]から[xmin, ymin, xmax, ymax]に
    boxes[:, :2] -= boxes[:, 2:] / 2 #座標(xmin, ymin)へ変換
    boxes[:, 2:] += boxes[:, :2] #座標(xmax, ymax)へ変換

    return boxes
