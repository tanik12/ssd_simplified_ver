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
from make_dataset import VOCDataset

def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば(3, 5)など変化します。
    この変化に対応したDataLoaderを作成するために、カスタマイズしたcollate_fnを作成します。
    collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    minibatch分の画像が並んでいるリスト変数batchに、
    minibatch番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0]) #sample[0]は画像imgです.
        targets.append(torch.FloatTensor(sample[1])) #sample[1]はアノテーション
                                                     #gtです。
    #imgsはminibatch sizeのリストになっている。
    #リストの要素はtorch.Size([3, 300, 300])です。
    #このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します。
    imgs = torch.stack(imgs, dim=0)

    #targetsはアノテーションデータの正解であるgtのリストです。
    #リストのサイズはminibatch sizeです。
    #リストtargetsの要素は[n, 5]となっています.
    #nは画像ごとに異なり、画像内にある物体の数になります。
    #5は[xmin, ymin, xmax, ymax, class_index]です。

    return imgs, targets

if __name__ == "__main__":
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

    ###
    batch_size = 32

    train_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

    #辞書型変数にまとめる
    dataloaders_dict = {"train" : train_dataloader, "val" : val_dataloader}

    #動作確認
    batch_iterator = iter(dataloaders_dict["val"]) #イテレータに変換
    images, targets = next(batch_iterator) #１番目の要素を取り出す
    print(images.size()) #torch.Size([4, 3, 300, 300])
    print(len(targets))
    print(targets[1].size()) #minibatchのサイズのリスト、各要素は[n, 5]、nは物体数

    print("train_datasetの要素数：", train_dataset.__len__())
    print("val_datasetの要素数：", val_dataset.__len__())
