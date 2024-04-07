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
param_grid_lr = {'C': [0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039]}

grid_search_lr = GridSearchCV(LogisticRegression(penalty='l2', max_iter=50000), param_grid_lr)
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
