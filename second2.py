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

model1 = load_model('cat_or_dog_main.h5')
model2 = load_model('cat_or_dog_main_2.h5')
model3 = load_model('cat_or_dog_main_3.h5')
model4 = load_model('cat_or_dog_main_4.h5')

i_size = 100

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('whatever_you_want/test_set', target_size=(i_size, i_size))
# test_set = test_datagen.flow_from_directory('google', target_size=(i_size, i_size))

ran = random.randint(0, 2000)
# ran = 19

test_image = image.load_img('whatever_you_want/test_set/{}'.format(test_set.filenames[ran]), target_size=(100, 100))
# test_image = image.load_img('google/{}'.format(test_set.filenames[ran]), target_size=(100, 100))

print(('whatever_you_want/test_set/{}'.format(test_set.filenames[ran])))
# print('google/{}'.format(test_set.filenames[ran]))
test_image = image.img_to_array(test_image)

test_image /= 255
test_image = test_image.astype('float32')

test_image = np.expand_dims(test_image, axis=0)

predictions1 = model1.predict(test_image)
predictions2 = model2.predict(test_image)
predictions3 = model3.predict(test_image)
predictions4 = model4.predict(test_image)

txt1 = 'model1'
txt2 = 'model2'
txt3 = 'model3'
txt4 = 'model4'

# img = Image.open('google/{}'.format(test_set.filenames[ran]))
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot()
# ax.imshow(img, cmap=plt.cm.binary)
# plt.show()


def proverka(predictions, txt):
    print(txt)
    if predictions[0, 0] > predictions[0, 1]:
        print("                     ")
        print("  ▕▔╲▂▂╱▔▏  Я кошка  ")
        print("  ╱╭╮┈┈╭╮╲  на {}%   ".format(int(predictions[0, 0] * 100)))
        print("  ▏┊▋┈▃┊▋┈▏  /       ")
        print("  ▏┈┏┳┻┳┓▕  /        ")
        print("  ▏┈┃╭━╮┃▕           ")
        print("  ╲┈╰┻━┻╯╱           ")
        print("   ▔▔▏▕▔▔            ")
        print("                     ")

    else:
        print("                          ")
        print(" ╭╮ ╱▔╲▂▂▂╱▔╲  Я собака   ")
        print(" ┃┃▕▕╲╭┈╮╭┈╱▏▏ на {}%     ".format(int(predictions[0, 1] * 100)))
        print(" ┃┃▕╱▕┊▋┊┊▋▏╲▏  /         ")
        print(" ┃▔▔▔▔╰━╯╰━▇╮  /          ")
        print(" ┃     ╰┳┳┳━╯             ")
        print(" ┃┣━┫┣┫┃╰━╯               ")
        print(" ╰╯ ╰╯╰╯                  ")
        print("                          ")


proverka(predictions1, txt1)
proverka(predictions2, txt2)
proverka(predictions3, txt3)
proverka(predictions4, txt4)
