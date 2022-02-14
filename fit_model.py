import numpy as np
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D)

def GenreModel(input_shape = (288, 432, 4), classes = 10):
  np.random.seed(10)
  X_input = Input(input_shape)

  X = Conv2D(8, kernel_size = (3, 3), strides = (1, 1), kernel_initializer = glorot_uniform(seed = 10))(X_input)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(16, kernel_size = (3, 3), strides = (1, 1), kernel_initializer = glorot_uniform(seed = 10))(X)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(32, kernel_size = (3, 3), strides = (1, 1), kernel_initializer = glorot_uniform(seed = 10))(X)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(64, kernel_size = (3, 3), strides = (1, 1), kernel_initializer = glorot_uniform(seed = 10))(X)
  X = BatchNormalization(axis = -1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Flatten()(X)

  X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed = 10))(X)

  model = Model(inputs = X_input, outputs = X, name = 'GenreModel')

  return model

def main():
  train_path = './data/train'
  train_datagen = ImageDataGenerator(rescale = 1./255)
  train_generator = train_datagen.flow_from_directory(train_path, target_size = (288, 432), color_mode = "rgba", class_mode = 'categorical', batch_size = 32)

  test_path = './data/test'
  test_datagen = ImageDataGenerator(rescale = 1./255)
  test_generator = test_datagen.flow_from_directory(test_path, target_size = (288, 432), color_mode = 'rgba', class_mode = 'categorical', batch_size = 32)

  model = GenreModel(input_shape = (288, 432, 4), classes = 10)
  opt = Adam(learning_rate = 0.00005)

  model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

  history = model.fit_generator(train_generator, epochs = 40, validation_data = test_generator)

  model.save("my_model.h5")
  
  with open('historyDict.txt', 'w') as file:
  	file.write(str(history.history))
 	
main()
