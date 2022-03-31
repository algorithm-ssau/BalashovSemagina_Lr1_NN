import random

from keras.preprocessing.image import ImageDataGenerator
import pydot
import graphviz
import numpy as np
import scipy
import image
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from PIL import Image

model = load_model('cat_or_dog_main.h5')

# print(model.summary())
# tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)  # !!!!!!!!!!!!!

i_size = 100

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('whatever_you_want/test_set', target_size=(i_size, i_size))
# test_set = test_datagen.flow_from_directory('google', target_size=(i_size, i_size))

# ran = random.randint(0, 20)
ran = 19

class_names = ['Cat', 'Dog']

test_image = image.load_img('whatever_you_want/test_set/{}'.format(test_set.filenames[ran]), target_size=(100, 100))
# test_image = image.load_img('google/{}'.format(test_set.filenames[ran]), target_size=(100, 100))

test_labels = test_set.classes

# print(('whatever_you_want/test_set/{}'.format(test_set.filenames[ran])))
test_image = image.img_to_array(test_image)

test_image /= 255
test_image = test_image.astype('float32')

test_image = np.expand_dims(test_image, axis=0)

predictions = model.predict(test_image)

img = Image.open('whatever_you_want/test_set/{}'.format(test_set.filenames[ran]))
# img = Image.open('google/{}'.format(test_set.filenames[ran]))


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array = [predictions_array[0, 0], predictions_array[0, 1]]
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777", width=0.6)
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(ran, predictions, test_labels, img)
plt.subplot(1, 2, 2)
plot_value_array(ran, predictions, test_labels)
plt.show()
