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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
df = df.dropna()  # Drop rows with missing values

X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter Tuning for Logistic Regression
param_grid_lr = {'C': [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.6, 0.065, 0.07, 0.075]}

# Initialize Logistic Regression with class weights
lr_model = LogisticRegression(penalty='l2', max_iter=500000, class_weight=dict(zip(np.unique(y), class_weights)))

grid_search_lr = GridSearchCV(lr_model, param_grid_lr)
grid_search_lr.fit(X_scaled, y)

# Get the best Logistic Regression model
best_model_lr = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Logistic Regression model:", best_model_lr)

# Save the best model
with open("model.pkl", 'wb') as f:
    pickle.dump(best_model_lr, f)
