import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('waferImg26x26.pkl')
images = df.images.values
labels = df.labels.values
labels = np.asarray([str(l[0]) for l in labels])

found_labels = {}

for i in range(len(images)):
    label = labels[i]
    if label not in found_labels:
        found_labels[label] = []
    found_labels[label].append(images[i])

# Determine the number of unique labels
num_unique_labels = len(found_labels)

# Create a subplot grid with a 3x3 layout
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for idx, label in enumerate(found_labels):
    row, col = divmod(idx, 3)
    image = found_labels[label][0]
    ax = axes[row, col]
    ax.imshow(image.transpose(1, 2, 0))
    ax.set_title(label)

# Remove any empty subplots
for idx in range(num_unique_labels, 9):
    row, col = divmod(idx, 3)
    fig.delaxes(axes[row, col])

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
