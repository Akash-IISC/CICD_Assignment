"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #added standardscaler
from sklearn.model_selection import GridSearchCV #added grid search
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
df = df.dropna()  # Drop rows with missing values

X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Hyperparameter Tuning for Logistic Regression
param_grid_lr = {'C': [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.6, 0.065, 0.07, 0.075]}

grid_search_lr = GridSearchCV(LogisticRegression(penalty='l2', max_iter=500000), param_grid_lr)
grid_search_lr.fit(X_scaled, y)

# Get the best Logistic Regression model
best_model_lr = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Logistic Regression model:", best_model_lr)


# Save the best model
with open("model.pkl", 'wb') as f:
    pickle.dump(best_model_lr, f)

#with open("model.pkl", 'wb') as f:
 #   pickle.dump(model, f)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# Load the dataset
df = pd.read_csv("data/train.csv")
df = df.dropna()  # Drop rows with missing values

X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Create Logistic Regression model with class weighting
model = LogisticRegression(class_weight={0: class_weights[0], 1: class_weights[1]}, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the trained model
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
