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

model = load_model('cat_or_dog_main_4.h5')

# print(model.summary())
# tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)  # !!!!!!!!!!!!!
n = 1000
result_set = []
i_size = 100
class_names = ['Cat', 'Dog']
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('whatever_you_want/test_set', target_size=(i_size, i_size))
test_labels = test_set.classes


def reboot(j):
    for i in range(j):
        ran = random.randint(0, 1999)
        test_image = image.load_img('whatever_you_want/test_set/{}'.format(test_set.filenames[ran]),
                                    target_size=(i_size, i_size))
        test_image = image.img_to_array(test_image)
        test_image /= 255
        test_image = test_image.astype('float32')
        test_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(test_image)
        true_label = test_labels[ran]
        predicted_label = np.argmax(predictions)
        if predicted_label == true_label:
            result_set.append(1)
        else:
            result_set.append(0)
    return result_set


result = reboot(n)
print(result.count(1))
result = int(result.count(1) / n * 100)
print(result, "%")
