import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load the dataset
file_path = '/Users/divya/Desktop/ DAPM Charts /Final_Preprocessed_data.csv'  
data = pd.read_csv(file_path)

# Encoding the target variable if it's categorical
label_encoder = LabelEncoder()
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

# Splitting the dataset into features (X) and target variable (y)
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Splitting the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicting on both the train and test sets
y_pred_train = rf_classifier.predict(X_train)
y_pred_test = rf_classifier.predict(X_test)

# Calculating metrics for the train set
accuracy_train = accuracy_score(y_train, y_pred_train)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
class_report_train = classification_report(y_train, y_pred_train)

# Calculating metrics for the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

# Print the metrics
print("Training Metrics:")
print("Accuracy:", accuracy_train)
print("Classification Report:\n", class_report_train)

print("\nTesting Metrics:")
print("Accuracy:", accuracy_test)
print("Classification Report:\n", class_report_test)

# Function to plot Confusion Matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({title})')

# Plotting confusion matrices
plot_confusion_matrix(conf_matrix_train, 'Train')
plot_confusion_matrix(conf_matrix_test, 'Test')

# Function to plot ROC Curve
def plot_roc_curve(y_true, y_pred_prob, label=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{label} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

# Predict probabilities for ROC curve
y_pred_prob_train = rf_classifier.predict_proba(X_train)[:, 1]
y_pred_prob_test = rf_classifier.predict_proba(X_test)[:, 1]

# Plot ROC Curve for both train and test sets
plt.figure(figsize=(8, 6))
plot_roc_curve(y_train, y_pred_prob_train, 'Train')
plot_roc_curve(y_test, y_pred_prob_test, 'Test')
plt.legend()
plt.show()
