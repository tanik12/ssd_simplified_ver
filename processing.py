#フォルダ「utils」にあるdata_augumentation.pyからimport
#入力画像の前処理をするクラス
from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords,\
PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

from get_anno_data import Anno_xml2list
from get_datapath import make_datapath_list
import cv2
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練と推論で異なる動作をする。
    画像サイズを300×300にする。
    学習時はデータオーギュメントする。

    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (B, G, R)
        各色チャネルの平均値。
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
                'train' : Compose([
                    ConvertFromInts(),  # intをfloat32に変換
                    ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
                    PhotometricDistort(),  # 画像の色調などをランダムに変化
                    Expand(color_mean),  # 画像のキャンバスを広げる
                    RandomSampleCrop(),  # 画像内の部分をランダムに抜き出す
                    RandomMirror(),  # 画像を反転させる
                    ToPercentCoords(),  # アノテーションデータを0-1に規格化
                    Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
                    SubtractMeans(color_mean)  # BGRの色の平均値を引き算
                ]),
                'val' : Compose([
                    ConvertFromInts(), #intをfloat32に変換
                    Resize(input_size), #画像サイズをinput_size×input_sizeに変換
                    SubtractMeans(color_mean) #BGRの色の平均を引き算
                ])
            }

    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """

        return self.data_transform[phase](img, boxes, labels)

if __name__ == "__main__":
    #動作確認
    #画像の読み込み
    rootpath = "/Users/gisen/git/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
    image_file_path = train_img_list[0]
    img = cv2.imread(image_file_path) #[高さ][幅][BGR]
    height, width, channels = img.shape #画像のサイズを取得
    
    #アノテーションをリストに
    voc_classes = ['aeroplane','bicycle','bird','boat','bottle','bus',\
                   'car','cat','chair','cow','diningtable','dog','horse',\
                   'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    transform_anno = Anno_xml2list(voc_classes)
    anno_list = transform_anno(train_anno_list[0], width, height)

    #元画像の表示
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    #前処理クラスの作成
    color_mean = (104, 117, 123) #(BGR)の色の平均値
    #color_mean = (0.485, 0.456, 0.406)
    input_size = 300 #画像のinputサイズを300×300にする
    transform = DataTransform(input_size, color_mean)

    #train画像の表示
    phase = 'train'
    img_transformed, boxes, labels = transform(
            img, phase, anno_list[:, :4], anno_list[:, :4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

    #val画像の表示
    phase = 'val'
    img_transformed, boxes, labels = transform(
            img, phase, anno_list[:, :4], anno_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()
