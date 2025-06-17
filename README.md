# CodeSoft (BATCH29)
## Data Science Internship
This repository contains three models built for classification and regression tasks using popular datasets. For my DataScience Internship

---

## Projects
### 1. Titanic Survival Prediction
Predict whether a passenger survived the Titanic disaster based on features like:
> Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

- Algorithm: Random Forest Classifier
- Output: titanic_model.pkl



### 2️. Movie Rating Prediction
Predict movie ratings using various movie features.
> Name, Year, Duration, Genre, Rating, Votes, Director, Actors

- Algorithm: Linear Regression / Random Forest Regressor (as per your code)
- Output: movie_rating_model.pkl



### 3️. Iris Flower Classification
Classify Iris flowers into:

> Iris-setosa,
> Iris-versicolor,
> Iris-virginica

based on:

> sepal_length, sepal_width, petal_length, petal_width

- Algorithm: Random Forest Classifier
- Output: iris_model.pkl

---


## How to Run
For each model:
- 1️. Load the dataset
- 2️. Preprocess the data (handle missing values, encode categories)
- 3️. Train the model
- 4️. Evaluate performance
- 5️. Save the model using joblib

---

## Requirements
see requirements.txt:
- pandas
- scikit-learn
- joblib

##### Install dependencies:
pip install -r requirements.txt

----
## Files
- titanic_model.pkl — Trained Titanic survival prediction model

- movie_rating_model.pkl — Trained movie rating prediction model

- iris_model.pkl — Trained Iris flower classification model

- README.md — Documentation

- requirements.txt — Dependencies

----
## Note
These models are designed as starter machine learning projects suitable for beginners learning model training, evaluation, and saving/loading with scikit-learn.


Example: Load a saved model for prediction
```python
import joblib

model = joblib.load('iris_model.pkl')
# example input: [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
print(prediction)
```
---

## Credits
- Project by: *NIYATI*
 - Developed as part of CodeSoft internship projects.
