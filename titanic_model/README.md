Titanic Survival Prediction

This project uses the Titanic dataset to predict passenger survival using a Random Forest Classifier.  
The model is trained, evaluated, and saved for future use.

* Files included

File
Description
`titanic_survival.py`
Python code for loading data, preprocessing, training, evaluation, and saving the model
`titanic_survival_model.pkl`
The trained Random Forest model
`requirements.txt`
List of Python dependencies
`README.md`
Project overview

* Dataset
We use the Titanic training dataset (`train.csv`) with features:
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Sex`: Gender
- `Age`: Age in years
- `SibSp`: # of siblings / spouses aboard
- `Parch`: # of parents / children aboard
- `Fare`: Passenger fare
- `Embarked`: Port of embarkation

* Model
- Algorithm: Random Forest Classifier  
- Library: `scikit-learn`
- Evaluation: Accuracy, confusion matrix, classification report, feature importance plot  

* How to run

1. Clone this repo:
git clone https://github.com/niyati666/titanic-survival-prediction.git cd titanic-survival-prediction
2. Install dependencies:
pip install -r requirements.txt
3. Run the model code:
python titanic_survival.py
* Outputs
1. Model test accuracy
2. Confusion matrix + classification report
3. Feature importance plot
4. Saved model file: titanic_survival_model.pkl

* Requirements
See requirements.txt.

* Credits
Project by: NIYATI
Dataset: Kaggle Titanic Dataset

