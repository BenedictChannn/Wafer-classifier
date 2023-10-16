import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from keras import layers, Input, models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_pickle('waferImg26x26.pkl')
images = df.images.values
labels = df.labels.values
labels = np.asarray([str(l[0]) for l in labels])

# Find the indices of images with the 'Donut' label to remove since only 1 image, hard to process
donut_indices = [i for i, label in enumerate(labels) if label == 'Donut']
images = np.delete(images, donut_indices)
labels = np.delete(labels, donut_indices)

# Create a mapping from string labels to integers
label_to_int = {label: i for i, label in enumerate(list(set(labels)))}

# Convert string labels to integers
labels = np.array([label_to_int[label] for label in labels], dtype=np.int32)
print(labels)

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
hidden = layers.Dense(64, activation='relu', name='hidden_1')(inputs)
hidden = layers.Dropout(0.2)(hidden)
hidden = layers.Dense(64, activation='relu', name='hidden_2')(hidden)
hidden = layers.Dropout(0.2)(hidden)

num_classes = len(set(labels))
outputs = layers.Dense(num_classes, activation='softmax', name='prediction')(hidden)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

model = keras.Model(inputs, outputs)
model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate=0.0001),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = [keras.metrics.SparseCategoricalAccuracy()],
)

# Fit model on training data, with validation at each epoch
history = model.fit(
    x_train, y_train, 
    batch_size=64, epochs=50, validation_data=(x_vali, y_vali),
    class_weight=class_weight_dict)

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

# Determine confusion matrix
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = y_test.astype('int')
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Calculate the recall score
recall = tf.keras.metrics.Recall()
recall.update_state(y_test, y_pred)
print("Test recall score: ", recall.result().numpy())

# Present confusion matrix in a more visual manner using seaborn
classes = list(label_to_int.keys())
plt.figure()
sns.set(font_scale=1)
heat_map = sns.heatmap(
    matrix, annot=True, annot_kws={'size':5}, cmap='YlGnBu', 
    linewidths=0.2, xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
