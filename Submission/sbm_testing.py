# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
from model import model


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


import pandas as pd
dataset = pd.read_csv('Training.csv')
print(dataset.shape)
dataset.info()

X = dataset.to_numpy()
X = tf.convert_to_tensor(X)

m = model('')
p = m.predict(X)
