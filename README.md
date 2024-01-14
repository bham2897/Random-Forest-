# Random-Forest-

**Overview**

The script utilizes a Random Forest Classifier to predict disease diagnoses based on various features. It includes steps for data preprocessing, model training, performance evaluation, and result visualization. The goal is to develop a robust model that can accurately classify disease status.

**Dataset**

The script operates on Final_Preprocessed_data.csv, which contains relevant features for disease diagnosis prediction. The dataset undergoes a splitting process into training and testing sets to assess the model's performance.

**Model**

The Random Forest Classifier, a powerful ensemble learning method, is used for its capability to handle large datasets and provide accurate predictions. It is known for its high accuracy and ability to run efficiently on large datasets.

**Performance Metrics**

Model performance is evaluated using several metrics:
Accuracy
Confusion Matrix
Classification Report
ROC (Receiver Operating Characteristic) Curve
AUC (Area Under the Curve)


**Visualization**
Visualizations included in the script are:

Confusion Matrix: Highlights the true positive, true negative, false positive, and false negative counts.

ROC Curve: Illustrates the diagnostic ability of the classifier, comparing the true positive rate to the false positive rate.
Preprocessing

The target variable is encoded for compatibility with the model. Features are selected based on their relevance to the disease diagnosis.

**Model Training and Evaluation**

The Random Forest model is trained on the training set and evaluated on both the training and testing sets. Metrics such as accuracy, confusion matrix, and classification report are computed to understand the model's performance.

**Requirements**

Essential libraries for this script include pandas, numpy, matplotlib, seaborn, and scikit-learn, which are vital for data manipulation, model training, and visualization.

