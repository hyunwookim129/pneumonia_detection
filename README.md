# pneumonia_detection

Title: Pneumonia classification using chest X-ray images

author: Hyun Woo Kim
code: https://github.com/hyunwookim129/pneumonia_detection

Introduction
This report summarizes the development, training, and evaluation of a deep learning model designed to classify chest X-ray images into two categories: Normal and Pneumonia. The dataset, obtained from Kaggle, consists of an imbalanced distribution of images, which presented challenges that were addressed through data augmentation.
Dataset Description

Training Set:
Total images: 5216
Class distribution:
Normal : 1341
Pneumonia: 3875
The imbalance between NORMAL and Pneumoniaclasses required special handling to prevent bias toward the majority class.
Validation Set:
Total images: 16
Class distribution: Approximately equal across classes.
The small size of the validation set may cause fluctuations in validation loss and accuracy, which is why redistributing the training and validation sets was recommended.

Test Set:
Total images: 624
Class distribution:
Normal : 234
Pneumonia: 390


The training and validation sets were not reallocated, and class weights were not applied during training. This was an intentional choice to assess the model's ability to handle the dataset as given. While this approach provides insights into the model's raw performance, future iterations could benefit from rebalancing the dataset or using class weights to mitigate bias


Data Augmentation

To enhance generalization and mitigate overfitting, the training data was augmented using the following transformations:
Rescaling: Normalize pixel values to [0, 1].
Rotation: Up to 20 degrees.
Shifting: Horizontal and vertical shifts by up to 20%.
Shearing: Shear intensity of 20%.
Zooming: Random zoom by up to 20%.
Horizontal Flipping: Introduced variations in image orientation.

Model Architecture

The model used is a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras, with the following layers:
Three convolutional layers (Conv2D) that progressively increase the number of filters (32, 64, 128) and MaxPooling Layers:
Extract spatial features from the images.
Dropout Layer: Prevents overfitting by randomly dropping connections.
Fully Connected Layers: Dense(128, activation='relu') for high-level feature extraction.
Dense(1, activation='sigmoid') for binary classification.

Training Process

Loss Function: Binary Crossentropy.
Optimizer: Adam.
Metrics: Accuracy.
Early Stopping: Implemented to monitor validation loss and prevent overfitting with a patience of 5 epochs.
Batch Size: 32.

Results
1. Training Performance
Final Training Accuracy: 92.51%
Final Validation Accuracy: 81.25%
Final Training Loss: 0.1768
Final Validation Loss: 0.3404


2. Evaluation Metrics
Classification Report:
Metric  Normal  Pneumonia  Macro Avg  Weighted Avg
Precision  0.90   0.89      0.90         0.89
Recall     0.81   0.95      0.88         0.89
F1-Score   0.85   0.92      0.88         0.89

Overall Accuracy: 89%
AUC-ROC: 0.96
AUC-PR: 0.97

3. Detailed Metric Analysis
Normal Class:
Precision: 90% (90% of predicted Normal cases are correct).
Recall: 81% (81% of actual Normal cases are identified).
Indicates more false negatives (misclassifying Normal as Pneumonia).
Pneumonia Class:
Precision: 89% (89% of predicted Pneumonia cases are correct).
Recall: 95% (95% of actual Pneumonia cases are identified).
Excellent detection with very few missed cases.

ROC Curve:

AUC: 0.96.
Indicates excellent discrimination between classes, with high sensitivity and low false positive rates across thresholds.

Precision-Recall Curve:

AUC-PR: 0.97.
Confirms strong performance, especially for the minority class, ensuring high recall without compromising precision.





Calibration Curve:

The model is well-calibrated, with predicted probabilities aligning closely to true probabilities.

Strengths of the Model
High Sensitivity for Pneumonia:
Recall of 95% ensures very few Pneumonia cases are missed.
Strong Calibration:
Predicted probabilities are reliable, making the model suitable for applications requiring probabilistic outputs.
Balanced Performance:
High AUC-ROC and AUC-PR demonstrate the model's ability to handle imbalanced data effectively. 
Limitations

Class Imbalance:
The higher number of Pneumonia images compared to Normal images could still cause bias toward predicting Pneumonia.
Small Validation Set:
The validation set size (16 images) is too small, potentially causing fluctuations in validation metrics.
Validation Accuracy:
Validation accuracy (81.25%) is lower than test accuracy (89%), suggesting potential overfitting to the training set.

Recommendations for Improvement
Redistribute Training and Validation Sets:
Combine the current training and validation sets, then redistribute into 80% training and 20% validation.
Class Balancing:
Use techniques such as oversampling the NORMAL class or applying class weights in the loss function to reduce bias.

Model Refinement:
Experiment with deeper architectures (e.g., ResNet, EfficientNet).
Hyperparameter Tuning:
Adjust batch size, learning rate, and dropout rates to optimize performance.
Data Augmentation:
Continue leveraging augmentation to reduce overfitting and increase generalization.

Future Directions for Explainability
Integrate Grad-CAM:
Generate heatmaps to highlight regions of X-rays that the model focuses on for predictions.
Visualize Filters and Feature Maps:
Show what features the model learns at each layer and how it processes specific images.

LIME or SHAP:
Use these tools to explain individual predictions by identifying pixel or region contributions.

Attention Mechanisms:
Add attention layers to focus on relevant image regions, with visual outputs for interpretation.

Interactive Dashboard:
Create a tool to combine Grad-CAM, confidence scores, and prediction probabilities for clinician-friendly insights.

Bias Evaluation:
Test the model across diverse patient demographics and imaging conditions to ensure fairness.
These steps will enhance transparency and trust, making the model more suitable for medical applications. Let me know if you'd like assistance implementing these!

Conclusion
The model demonstrates strong performance in distinguishing between normal and Pneumonia cases, with high accuracy, sensitivity, and precision. Its robustness is supported by excellent AUC-ROC and AUC-PR scores, and well-calibrated predictions. However, addressing dataset imbalance and validation set limitations can further enhance reliability and performance.

