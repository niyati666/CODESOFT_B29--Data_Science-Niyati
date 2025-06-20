# -*- coding: utf-8 -*-
"""Titanic Survival.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1quJbFyFS4ZTppDdtrRfpXFBzZWY-FPOx
"""

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from google.colab import files
uploaded = files.upload()

import pandas as pd
data = pd.read_csv('/content/train.csv')
data.head()

# Load Titanic dataset
data = pd.read_csv('train.csv')

# Show first 5 rows
print(data.head())

# Basic info
print(data.info())

# Checking missing values
print(data.isnull().sum())

# Filling missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Filling missing Embarked with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop Cabin (got too many missing values)
data.drop(columns=['Cabin'], inplace=True)

# Encode Sex (male:0, female:1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Encode Embarked (S:0, C:1, Q:2)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print(data[['Age', 'Embarked']].head())

# Features to use
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = data[features]
y = data['Survived']

# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Detailed report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot feature importance
importances = model.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance")
plt.show()