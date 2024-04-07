import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle

# Load data
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter Tuning for Logistic Regression
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 80, 100]}  # Adjust the range of 'C' as needed
grid_search_lr = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000), param_grid_lr)
grid_search_lr.fit(X_scaled, y)

# Get the best Logistic Regression model
best_lr_model = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Logistic Regression model:", best_lr_model)

# Save the best Logistic Regression model
with open("model.pkl", 'wb') as f:
    pickle.dump(best_lr_model, f)
