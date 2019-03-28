from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

import matplotlib
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

net = models.Sequential()
net.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
net.add(layers.MaxPooling2D((2, 2)))
net.add(layers.Conv2D(64, (3, 3), activation='relu'))
net.add(layers.MaxPooling2D((2, 2)))
net.add(layers.Conv2D(64, (3, 3), activation='relu'))
net.add(layers.Flatten())
net.add(layers.Dense(64, activation='relu'))
net.add(layers.Dense(10, activation='softmax'))

net.summary()

net.compile(optimizer='rmsprop',
    loss='categorical_crossentropy', 
    metrics=['accuracy']) 

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images  = test_images.reshape((10000, 28, 28, 1))
test_images  = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)

history = net.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

model_json_str = net.to_json()
open('mnistmodel.json', 'w').write(model_json_str)
net.save_weights('mnist_weights.h5');

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

