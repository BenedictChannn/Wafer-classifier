import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

df = pd.read_pickle('waferImg26x26.pkl')
images = df.images.values
labels = df.labels.values
labels = np.asarray([str(l[0]) for l in labels])

# Flatten the images
images_flat = np.array([image.reshape(-1) for image in images])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(images_flat, labels, test_size = 0.2, random_state=24)


# Set hyperparameters to be tuned
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

param_search = GridSearchCV(LogisticRegression(max_iter = 1000), param_grid, cv = 5) # Set to 10-fold cross validation
param_search.fit(x_train, y_train)
print(param_search.best_params_)



# # Find how many of each image in each label
# label_counts = Counter(labels)

# # Print the label counts
# for label, count in label_counts.items():
#     print(f"Label '{label}': Count = {count}")

# for image in images:
#     plt.imshow(image.transpose(1, 2, 0))
#     plt.show()