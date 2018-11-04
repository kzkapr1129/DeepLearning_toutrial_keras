from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

import matplotlib
import matplotlib.pyplot as plt

'''
CNN(Convolutional Neural Network)とは？
  mnist1のような全結合のみのネットワークの上に畳み込み層と呼ばれるものを追加したニューラルネットワーク。
  畳み込み層とはフィルターと呼ばれるものを使用して画像の特徴を出力することを目的とした層。
  全結合層のみでは画像中の全ピクセルで学習するのに対して、畳み込み層はエッジ情報を全結合層に入力するため正解率が向上することが期待できる。
  畳み込みはConv2D()とMaxPooling2D()のセットで使われ、Conv2D()でフィルター処理をして、MaxPooling2D()でN*Nピクセルを1ピクセルに情報削減して出力し効率的にデータ削減・学習を行う。
  欠点は全結合層のみと比較して沢山の学習時間が必要になること。
  学習済みネットワークを再利用したりファインチューニングとよばれる方法を使うことで学習時間を減らすことも可能。
'''

# MINISTの訓練用・テスト用画像(28x28)の配列とそのラベル(1が描画された画像なら'1')配列を読み込む
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# ニューラルネットワークの構築
net = models.Sequential()

# 畳み込み層を追加する
# 3x3の32個のフィルターで畳み込み演算を行う
net.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# pooling層を追加する
# 2x2ピクセルを1ピクセルに情報削減する
net.add(layers.MaxPooling2D((2, 2)))

# 畳み込み層を追加する
# 3x3の64個のフィルターで畳み込み演算を行う
net.add(layers.Conv2D(64, (3, 3), activation='relu'))

# pooling層を追加する
# 2x2ピクセルを1ピクセルに情報削減する
net.add(layers.MaxPooling2D((2, 2)))

# 畳み込み層を追加する
# 3x3の64個のフィルターで畳み込み演算を行う
net.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 入力データを1次元に平坦化して出力する
# 畳み込み層から全結合層をつなげるときは、必ずこの処理が必要
net.add(layers.Flatten())

# 全結合層を追加する
net.add(layers.Dense(64, activation='relu'))

# 出力層(softmax)を追加する
net.add(layers.Dense(10, activation='softmax'))

# 作成したニューラルネットワークをプロットする
net.summary()

# ニューラルネットワークをコンパイルする
net.compile(optimizer='rmsprop',     # オプティマイザ(損失関数の結果に基づいてネットワークの重みバイアスを更新する)にrmspropを使用する
    loss='categorical_crossentropy', # 多クラス分類の場合は交差エントロピー(categorical_crossentropy)を損失関数に指定する
    metrics=['accuracy'])            # 訓練とテストの指標。accuracyを指標にする

# 訓練画像の前処理を行う
train_images = train_images.reshape((60000, 28, 28, 1)) # (60000, 28, 28)=(画像枚数, 画像の幅, 画像の高さ) から (60000, 28, 28, 1)=(画像枚数, 画像の幅, 画像の高さ、チャネル数)に変換する
train_images = train_images.astype('float32') / 255   # float型に変換し、0~1に正規化する。ネットワークの入力値はなるべく小さくする必要があるため

# テスト画像の前処理を行う
test_images  = test_images.reshape((10000, 28, 28, 1))
test_images  = test_images.astype('float32') / 255

# on-hotエンコードを行う(正解が1,それ以外が0の配列)
train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

# テスト出力 (0件目から20件目までを出力)
print(train_labels[0:20])

# 学習と検証を同時に行う
history = net.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

# 学習と検証の正解率・損失の推移を取得
acc = history.history['acc']          # 訓練データの正解率
val_acc = history.history['val_acc']  # 検証データの正解率
loss = history.history['loss']        # 訓練データの損失率
val_loss = history.history['val_loss']# 検証データの損失率

# グラフの縦軸(0~1), 横軸(1~5)
epochs = range(1, len(acc) + 1)

# 正解率をプロット
plt.plot(epochs, acc, 'b', label='[Training] accuracy')
plt.plot(epochs, val_acc, 'r', label='[Validation] accuracy')
plt.title('Accuracy') # グラフのタイトル
plt.legend() # 凡例のプロット

plt.figure()

# 損失率をプロット
plt.plot(epochs, loss, 'b', label='[Training] loss')
plt.plot(epochs, val_loss, 'r', label='[Validation] loss')
plt.title('Loss') # グラフのタイトル
plt.legend() # 凡例のプロット

# グラフの表示
plt.show()
