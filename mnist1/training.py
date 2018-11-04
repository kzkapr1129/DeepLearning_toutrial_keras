from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

import matplotlib
import matplotlib.pyplot as plt

# MINIST(エムニスト)の訓練用・テスト用画像(28x28)の配列とそのラベル(1が描画された画像なら'1')配列を読み込む
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# ニューラルネットワークの構築
net = models.Sequential()

# 1次元の28x28のベクトルを入力とする全結合層を追加する(入力層)
# パラメータの意味:
#   512-> ユニットの数。
#         512個のユニットを持っており、それぞれのユニットは「y = h(f(w1x1 + w2*x2 ・・・w784x784 + b))」という関数を実行している
#         h()は'活性化関数'、xは入力値。wとbは'重み'、'バイアス'で、学習を行うことで調整されていく。初期値は乱数が使用される
#   activation='relu' -> 活性化関数に'relu'を使用する
#         relu関数は入力が0未満なら0にし、以上ならその値を出力するというもの
#   input_shape -> この層に入力するテンソル(数値のコンテナ)の大きさを指定する
#         １画像が28x28のグレイスケールであるため、１次元の784ベクトルを入力として受け取る
net.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

# 隠れ層はなし
# net.add(layers.Dense....

# 出力層(入力画像が0~9の数字のどれに近いかを表す確率を出力する)を追加する
# パラメータの意味:
#   10 -> ユニットの数。0~9の数字の近さの確率が出力される
#   activation='softmax' -> 入力を0~1の値に収めて出力する活性化関数
# メモ:
#   input_shapeは省略されているので、前層のユニット数(512)が入力となるように自動でテンソルのサイズが決められる
net.add(layers.Dense(10, activation='softmax'))

# メモ: ユニット数について
#   ディープラーニングは層を深くすればするほど複雑な問題を解くことができるが、反面学習する必要があるパラメータの数が増えるため「過学習」という問題が起きやすくなる
#   対策として出力層に着く前にパラメータの数を減らす必要があり、今回は784->512->10の順で減らしていった
#   出力層が100必要になるような複雑な問題を解く場合は入力・隠れ層のユニット数・階層をもっと増やさないといけないかもしれない
#   その場合は過学習が起きやすくなるためドロップアウト、正則化、学習データ数の水増し等を行うことがある

# 作成したニューラルネットワークをプロットする
net.summary()

# ニューラルネットワークをコンパイルする
net.compile(optimizer='rmsprop',     # オプティマイザ(損失関数の結果に基づいてネットワークの重みバイアスを更新する)にrmspropを使用する
    loss='categorical_crossentropy', # 多クラス分類の場合は交差エントロピー(categorical_crossentropy)を損失関数に指定する
    metrics=['accuracy'])            # 訓練とテストの指標。accuracyを指標にする

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
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

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
