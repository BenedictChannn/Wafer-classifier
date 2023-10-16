# Automated Wafer Inspection using Convolutional Neural Network (CNN)

## Introduction

The automation of visual wafer inspection is a pivotal application in industrial automation. Traditionally, experts manually inspect wafers for defects, but automating this process can significantly improve efficiency and reduce human error. In this project, we leverage the power of Convolutional Neural Networks (CNNs) to automate wafer inspection.

## Dataset Analysis

The provided dataset consists of 14,366 elements, each with an image and a classification label. These images are in the format (3, 26, 26), signifying 26x26 pixel RGB images. The dataset is distributed across nine classes, representing various wafer anomalies:

1. 'None' (13,489 images): Defect-free wafers.
2. 'Edge-Loc' (296 images): Wafers with edge-related defects.
3. 'Loc' (297 images): Wafers with localized defects.
4. 'Center' (90 images): Wafers with center anomalies.
5. 'Scratch' (72 images): Wafers featuring scratches.
6. 'Random' (74 images): Wafers with randomly scattered anomalies.
7. 'Near-full' (16 images): Wafers that are near full, possibly lacking clear defects.
8. 'Edge-Ring' (31 images): Wafers with ring-shaped edge anomalies.
9. 'Donut' (1 image): Wafers with a circular defect at the center.

![Defects](https://github.com/BenedictChannn/Wafer-classifier/raw/main/Defects.png)


An immediate observation is the significant class imbalance. The 'None' class is abundant, while 'Donut' is underrepresented. Addressing this imbalance is crucial for accurate model training. To address this, we applied class weights to encourage the model to focus on underrepresented classes.

## Model Approach

We decided to employ a CNN for this task, as it excels in image classification. The model architecture consists of two hidden layers with ReLU activation functions. The dataset was split into training, validation, and test sets with 9050, 3879, and 1437 images, respectively.

To combat overfitting, dropout layers were added to each hidden layer. They diversify the model's learning process by randomly "dropping out" a fraction of neurons during training.

Hyperparameter tuning led us to the following values:
- Epochs: 50
- Learning rate: 0.0001
- Batch size: 64
- Dropout: 0.2

The model was trained, and we observed an improvement in accuracy and a reduction in loss over the epochs. The test set yielded a loss of 44.8% and an accuracy of 92%. A confusion matrix and a recall score of 99.6% demonstrated the model's effectiveness in identifying defective wafers.

![Confusion Matrix](https://github.com/BenedictChannn/Wafer-classifier/blob/main/Results/Confusion_matrix_Epoch50_lr0.0001_dropout0.2_weighted.png)


## Future Improvements

To further enhance accuracy and robustness, data augmentation with geometric transformations can be applied to create additional data. Additional data collection efforts for the 'Donut' class are also recommended to address its underrepresentation.

---

Feel free to explore the project code and further details in this repository. Your feedback and contributions are highly appreciated!
