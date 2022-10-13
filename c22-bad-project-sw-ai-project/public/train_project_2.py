# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

# 1. Load the data
data_dir = pathlib.Path("datasets")
# NOTE: We specify the classes we want to categorize into. We change the code of other models to fit into our own project. (i.e. Transfer Learning)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("image_count: ", image_count)
class_names = list(data_dir.glob('*/'))
print("class_names: ", class_names)

# 2. Devide the data into training dataset and validation dataset
# NOTE: we will follow the image height and width of the base model

BATCH_SIZE = 32
# NOTE: try to lower the BATCH_SIZE if the computer is not capable of loading too many input
IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
# "3," means each pixel has 3 values
IMG_SHAPE = IMG_SIZE + (3,)


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# 3. Normalize the data
# NOTE: during this process, we change the pixel value of image to the range of 0 to 1
# NOTE: we will follow the rescale value of the base model i.e.(1./127.5)-1
normalization_layer = tf.keras.layers.Rescaling((1./127.5)-1)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.4)
])

# the preporcess input of the moblenet_v2 (has its own data augmentation)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 4. Create a base model from a pre-trained model
# We need to know what are the
#   - inputs (i.e. (160,160,3))
#   - outputs (i.e. (5,5,1280))
# Then, we know how we can pre-process our current input before this transfer learning
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


feature_batch = base_model(image_batch)
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)


# Add a classification head
# NOTE: The dense layer will consist 3 nodes
prediction_layer = tf.keras.layers.Dense(len(class_names))
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
# NOTE: Dropout is a function for checking if the model is "hard-coding" (like a student reciting the answers of his previous quizzes, but not actually learning)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# Compile the model (the last preview before training)
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
# NOTE: We use "SparseCategoricalCrossentropy" (multiple classes) instead of "BinaryCrossentropy" (only two classes)
# NOTE: Forget about "BinaryCrossentropy". "SparseCategoricalCrossentropy" is also suitable when there are only two classes

# Start the training
# NOTE: epoch means the number of times of looping the SAME set of training dataset.
# (i.e. 操10次相同的past paper)
initial_epochs = 5

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds)


# NOTE: remember to save, or else the result will not be store and I need to re-train again next time.

model.save("mobile_net_v2_5epochs.h5", save_format='h5')
# NOTE: Another comment save_format is JSON, if I plan to use it in frontend. h5 cannot be loaded in frontend.
