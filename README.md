# Car-Price-Prediction
# Graduation Project

This repository contains the notebook **Graduation_Project_dqminh.ipynb**, which includes data analysis and machine learning experiments for the graduation project.

# 1.Import Libaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 2.Load data
df = pd.read_csv("CarPrice_Assignment.csv")
df.head()

df.info()
df.describe()

# 3.Pre-processing data
# Drop unnesscary values
df.drop(['car_ID', 'CarName'], axis=1, inplace=True)

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# 4.Feature Selection
X = df.drop("price", axis=1)
y = df["price"]

# Top 10 Feature
bestfeatures = SelectKBest(score_func=f_regression, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ["Specs", "Score"]
print(featureScores.nlargest(10, "Score"))

# 5.Train-test split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6.Train and evaluate models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

# XGBoost
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
print("XGBoost R²:", r2_score(y_test, y_pred_xgb))

# 7.Prediction
# Make predictions using the trained Random Forest model
y_pred = rf_model.predict(X_test)

# Display a sample of the predicted prices alongside actual prices
predictions_df = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

print(predictions_df.head())
