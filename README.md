# 🤖 AI Final Projects – Logistic Regression & AI Classifier Evaluation

Welcome to my Artificial Intelligence project repository! This contains two major classification-based Jupyter Notebooks and one project report, all focused on evaluating AI models using performance metrics and visualization techniques.

---

## 📂 Files Overview

- 📘 [AI Final Project.ipynb](https://github.com/Pranathivutla30/Artificial-intelligence/blob/main/AI%20Final%20Project.ipynb) – A full pipeline for churn prediction using logistic regression, preprocessing, scaling, evaluation metrics, and visualizations.

- 📄 [Project Report.pdf](https://github.com/Pranathivutla30/Artificial-intelligence/blob/main/Project%20Report.pdf) – A detailed summary report of the final project, including objectives, methodologies, and results.

- 📗 [second project.ipynb](https://github.com/Pranathivutla30/Artificial-intelligence/blob/main/second%20project.ipynb) – Model evaluation using ROC, learning curve, precision-recall curve, and confusion matrix.

---

## 🧠 Summary

- Built classification models using **Logistic Regression**
- Preprocessed real-world datasets (e.g., churn dataset)
- Evaluated model performance using:
  - Confusion Matrix
  - ROC Curve & AUC
  - Precision-Recall Curve
  - F1-Score, Precision, Recall
- Visualized results using **matplotlib** and **seaborn**
- Includes **Learning Curve Analysis** for overfitting/underfitting detection
- Documented entire workflow and findings in PDF report

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)
# Load the dataset
data = pd.read_csv('/content/churn.csv')

# Data Preprocessing
X = data.drop(['Churn'], axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Confusion Matrix
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ROC Curve
y_pred_prob = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5, scoring='accuracy')
