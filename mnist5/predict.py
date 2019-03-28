from keras.models import model_from_json
import cv2
import numpy
 
json_string = open('mnistmodel.json').read()
model = model_from_json(json_string)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('mnist_weights.h5')

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
input = numpy.array(img).reshape((28,28,1))
input = numpy.expand_dims(input, axis=0)
 
prediction = model.predict(input)
print("result={0}".format(numpy.argmax(prediction[0])))

