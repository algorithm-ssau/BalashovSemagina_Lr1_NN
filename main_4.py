from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   # rotation_range=40,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   # shear_range=0.2,
                                   # zoom_range=0.2,
                                   # horizontal_flip=True
                                   )

test_datagen = ImageDataGenerator(rescale=1. / 255)

i_size = 100
BATCH_SIZE = 100
i_range = 100

training_set = train_datagen.flow_from_directory('whatever_you_want/training_set',
                                                 target_size=(i_size, i_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('whatever_you_want/test_set',
                                            target_size=(i_size, i_size),
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(i_size, i_size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),

    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
    # tf.keras.layers.Dense(2)
])

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    training_set,
    steps_per_epoch=8000 / BATCH_SIZE,
    epochs=i_range,
    validation_data=test_set,
    validation_steps=2000 / BATCH_SIZE
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(i_range)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность на обучении')
plt.plot(epochs_range, val_acc, label='Точность на валидации')
plt.legend(loc='lower right')
plt.title('Точность')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери на обучении')
plt.plot(epochs_range, val_loss, label='Потери на валидации')
plt.legend(loc='upper right')
plt.title('Потери')
plt.savefig('./cat_or_dog_main_4.png')
plt.show()

model.save('cat_or_dog_main_4.h5')
