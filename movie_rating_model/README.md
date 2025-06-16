Movie Rating Prediction

This project builds a machine learning model to predict movie ratings based on features such as genre, director, and actors. The goal is to analyze historical movie data and accurately estimate ratings given by users or critics.


* Project Objectives
- Explore and analyze historical movie data.
- Preprocess categorical and numerical features.
- Build a regression model to predict movie ratings.
- Evaluate and save the trained model for future use.


* Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn


* Files
-movie_rating_model.pkl → Saved trained model (joblib format)
-rquirements.txt → List of dependencies
-README.md → Project documentation
-IMDb_Movies_India.csv → Dataset (if public and allowed by source)


* Dataset
The dataset contains features such as:
- Genre
- Director
- Actors
- Rating (target)


* Model
- Linear Regression
- Features encoded using `LabelEncoder`
- Evaluated using Mean Squared Error and R² Score


* How to Run
1. Clone the repo:
git clone https://github.com/niyati666/movie-rating-prediction.git cd movie-rating-prediction
2. Install dependencies:
pip install -r requirements.txt
3. Run the notebook:
-Open Google Colab or Jupyter Notebook.
-Upload and execute the provided notebook or Python script.


* Output
-R² Score
-Mean Squared Error
-Plot: Actual vs Predicted Ratings

* Future Improvements
-Use more advanced regressors (Random Forest, XGBoost)
-Add more features (budget, year, runtime)
-Hyperparameter tuning
-Web app integration (e.g., with Streamlit)


* Requirements
See requirements.txt

* Credits
Project by: NIYATI
Dataset: https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies
