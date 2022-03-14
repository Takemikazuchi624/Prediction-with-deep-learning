# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image
import pandas as pd
import openpyxl
import datetime
import csv

#print('Hello Koki')
#workbook = openpyxl.load_workbook("旅行スケジュール（原本）.xlsx")
#sheet = workbook.active
#answers = []
#print('Hello fukuoka')


def main(pd):
    # 入力画像のパラメータ
    img_width = 64 # 入力画像の幅
    img_height = 64 # 入力画像の高さ
    img_ch = 3 # 3ch画像（RGB）

    FILE_PATH = 'C:/Pythons/2.7.7 CLassification/ex1_data/sakuratest/'
    files = os.listdir(FILE_PATH)

    
    # データの保存先(自分の環境に応じて適宜変更)
    SAVE_DATA_DIR_PATH = "C:/Pythons/2.7.7 CLassification/ex1_data/"

    # ラベル
    labels =['さくらちゃん', 'りんご', '鯛']

    # 保存したモデル構造の読み込み
    model = model_from_json(open(SAVE_DATA_DIR_PATH + "model.json", 'r').read())

    # 保存した学習済みの重みを読み込み
    model.load_weights(SAVE_DATA_DIR_PATH + "weight.hdf5")

    # 画像の読み込み（32×32にリサイズ）
    # 正規化, 4次元配列に変換（モデルの入力が4次元なので合わせる）
    num = 0
    df = pd.read_csv('C:/Pythons/2.7.7 CLassification/ex1_data/sampleSolution.csv',index_col = 0)
    #row_index = 2

    for i in files:
        img = Image.open(os.path.join(FILE_PATH, i))
        img = img.convert("RGB")
        img = img.resize((img_width, img_height))  # リサイズ
        img = img_to_array(img) 
        img = img.astype('float32')/255.0
        img = np.array([img])

        # 分類機に入力データを与えて予測（出力：各クラスの予想確率）
        y_pred = model.predict(img)

        # 最も確率の高い要素番号
        number_pred = np.argmax(y_pred)
        #answers.append(number_pred)

        # 予測結果の表示
        print("y_pred:", y_pred)  # 出力値
        print("number_pred:", number_pred)  # 最も確率の高い要素番号
        print('label_pred：', labels[int(number_pred)]) # 予想ラベル
               
        
        #df = pd.read_csv('C:/Pythons/2.7.7 CLassification/ex1_data/sampleSolution.csv',index_col = 0)
        #print(df.iat[0,0])
        df.iat[num,0] = number_pred + 1
        num = num + 1
    df.to_csv('C:/Pythons/2.7.7 CLassification/ex1_data/sampleSolution.csv')
               
      


if __name__ == '__main__':
    main(pd)
