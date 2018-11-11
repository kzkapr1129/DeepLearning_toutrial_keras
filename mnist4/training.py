from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import backend as K

import matplotlib
import matplotlib.pyplot as plt

'''
MNIST(エムニスト)とは？
  0~9までの数字が描画された画像データ群。
  ディープラーニングを学習するときのHello Worldとして使用される
'''

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

# MINISTの訓練用・テスト用画像(28x28)の配列とそのラベル(1が描画された画像なら'1')配列を読み込む
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# ニューラルネットワークの構築
net = models.Sequential()
net.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
net.add(layers.Dense(10, activation='softmax'))

# 作成したニューラルネットワークをプロットする
net.summary()

# ニューラルネットワークをコンパイルする
net.compile(optimizer='rmsprop',   # オプティマイザ(損失関数の結果に基づいてネットワークの重みバイアスを更新する)にrmspropを使用する
    loss=categorical_crossentropy, # 自作の損失関数を適用する
    metrics=['accuracy'])          # 訓練とテストの指標。accuracyを指標にする

# 訓練画像の前処理を行う
train_images = train_images.reshape((60000, 28 * 28)) # (60000, 28, 28)=(画像枚数, 画像の幅, 画像の高さ) から (60000, 784)=(画像枚数, 画像サイズ)に変換する
train_images = train_images.astype('float32') / 255   # float型に変換し、0~1に正規化する。ネットワークの入力値はなるべく小さくする必要があるため

# テスト画像の前処理を行う
test_images  = test_images.reshape((10000, 28 * 28))
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
