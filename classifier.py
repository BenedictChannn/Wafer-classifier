import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_pickle('waferImg26x26.pkl')
images = df.images.values
labels = df.labels.values
labels = np.asarray([str(l[0]) for l in labels])


label_counts = Counter(labels)

# Print the label counts
for label, count in label_counts.items():
    print(f"Label '{label}': Count = {count}")

# for image in images:
#     plt.imshow(image.transpose(1, 2, 0))
#     plt.show()