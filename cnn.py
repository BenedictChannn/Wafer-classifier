import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, models
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

df = pd.read_pickle('waferImg26x26.pkl')
images = df.images.values
labels = df.labels.values
labels = np.asarray([str(l[0]) for l in labels])

# Create a mapping from string labels to integers
label_to_int = {label: i for i, label in enumerate(set(labels))}

# Convert string labels to integers
labels = np.array([label_to_int[label] for label in labels], dtype=np.int32)
print(labels)

print(df.dtypes)

# Flatten the images
images_flat = np.array([image.reshape(-1) for image in images])

x_train, x_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.1)
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.3)
print('Train x: {}, y : {}'.format(x_train.shape, y_train.shape))
print('Test x: {}, y : {}'.format(x_test.shape, y_test.shape))
print('Validation x: {}, y : {}'.format(x_vali.shape, y_vali.shape))

input_shape = images_flat[0].shape

inputs = Input(shape=input_shape, name="input_images")

# Add hidden layers
hidden = layers.Dense(32, activation='relu', name='hidden_1')(inputs)
hidden = layers.Dense(32, activation='relu', name='hidden_2')(hidden)

num_classes = len(set(labels))
outputs = layers.Dense(num_classes, activation='softmax', name='prediction')(hidden)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate=0.0001),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = [keras.metrics.SparseCategoricalAccuracy()],
)

# Fit model on training data, with validation at each epoch
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_vali, y_vali))

# Accuracy plot 
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# Evaluate on test set
results = model.evaluate(x_vali, y_vali, batch_size=32)
print("Test set loss: {}, Test set accuracy: {}".format(results[0], results[1]))