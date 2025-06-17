# Iris Flower Classification

This project uses the famous Iris dataset to train a machine learning model that classifies Iris flowers into three species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

The model is built using the Random Forest algorithm and uses the sepal and petal measurements as input features.


## Files
- iris_flower_model.pkl → Saved trained model (joblib format)
- rquirements.txt → List of dependencies
- README.md → Project documentation
- IRIS.csv → Dataset (if public and allowed by source)


## Dataset

The dataset contains 150 samples with 5 columns:
- sepal_length
- sepal_width
- petal_length
- petal_width
- species (target variable)

You can download the dataset from [Kaggle Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset).


## Model
- Algorithm: Random Forest Classifier
- Library: scikit-learn
- Input features: sepal length, sepal width, petal length, petal width
- Target: species (encoded using LabelEncoder)


## How to run the model

1. Load the dataset.
2. Preprocess the data by encoding species labels.
3. Split the data into training and testing sets.
4. Train the Random Forest Classifier.
5. Evaluate the model accuracy.
6. Save the trained model using joblib.


## Example code snippet

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv('Iris.csv')

# Features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'iris_model.pkl')
```

## How to Run

1. Clone the repo:
git clone https://github.com/your-username/iris-flower-classification.git cd iris-flower-classification

2. Install dependencies:
pip install -r requirements.txt

3. Run the notebook:
-Open Google Colab or Jupyter Notebook.
-Upload and execute the provided notebook or Python script.


## Result
The Random Forest model achieves high accuracy on the test data, successfully classifying Iris flower species based on their features.



## Requirements
See requirements.txt


## Credits
- Project by: NIYATI
- Dataset : https://www.kaggle.com/datasets/arshid/iris-flower-dataset
