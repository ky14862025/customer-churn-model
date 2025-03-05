# customer-churn-model
A simple AI model to predict customer churns in organizations
Customer Churn Prediction

Overview

This project focuses on predicting customer churn using machine learning and deep learning techniques. The dataset is preprocessed, analyzed, and used to train models to classify whether a customer is likely to churn.

Dependencies

Ensure you have the following libraries installed:

Python 3.x

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

TensorFlow/Keras

Keras Tuner

Pickle

Dataset

The dataset used for this project is CustomerChurn_dataset.csv. The customerID column is dropped, and missing values in TotalCharges are imputed.

Data Preprocessing

Converted TotalCharges to numeric and handled missing values.

Encoded categorical variables using LabelEncoder.

Standardized numerical features using StandardScaler.

Feature Selection

Used correlation analysis to identify top features.

Applied Recursive Feature Elimination (RFE) with RandomForestClassifier to select important features.

Exploratory Data Analysis (EDA)

Boxplots and count plots were generated to analyze numerical and categorical features with respect to churn.

Model Training

Machine Learning Model

Trained a RandomForestClassifier for feature selection.

Deep Learning Model

Implemented a Multi-Layer Perceptron (MLP) using Keras.

Used Keras Tuner for hyperparameter tuning.

Split data into training and test sets.

Optimized model using Adam optimizer and binary_crossentropy loss function.

Model Evaluation

Evaluated accuracy using test data.

Used roc_auc_score for performance analysis.

Saved the best-performing model as churn_assign.h5.

Saving Artifacts

The trained model is saved as churn_assign.h5.

The fitted StandardScaler is saved as scaler.pkl.

The LabelEncoder is saved as label_encoder.pkl.

How to Run

Ensure all dependencies are installed.

Load and preprocess the dataset.

Train the model using customer_churn_assignment.py.

Evaluate the model using the test set.

Save the trained model for future predictions.

Contact

For any questions or improvements, feel free to reach out!

