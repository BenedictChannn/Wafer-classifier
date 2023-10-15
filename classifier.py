import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('waferImg26x26.pkl')
images = df.images.values
labels = df.labels.values
labels = np.asarray([str(l[0]) for l in labels])

for label in labels:
    print(label)
# for image in images:
#     plt.imshow(image.transpose(1, 2, 0))
#     plt.show()