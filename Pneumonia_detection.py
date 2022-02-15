import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import os

from keras.preprocessing.image import ImageDataGenerator, load_img

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    vertical_flip = True,
    zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale = 1./255)


train_generator = train_datagen.flow_from_directory(
    'train',
    class_mode = 'binary',
    batch_size = 32,
    target_size = (256, 256)
)

test_generator = test_datagen.flow_from_directory(
    'test',
    class_mode = 'binary',
    batch_size = 32,
    target_size = (256, 256)
)

