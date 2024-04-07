import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #added standardscaler
from sklearn.model_selection import GridSearchCV #added grid search
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter Tuning with GridSearchCV
param_grid = {'C': [0.5, 0.1, 0.01]}  # Values of 'C' to try
grid_search = GridSearchCV(LogisticRegression(penalty='l2', max_iter=2000), param_grid)
grid_search.fit(X_scaled, y)

# Get the best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best model:", best_model)

#model = LogisticRegression().fit(X, y)
#model = LogisticRegression(max_iter=2000).fit(X, y)
#model = LogisticRegression(penalty='l2', C=1.0, max_iter=2000).fit(X_scaled, y) #regularization with scaling

# Save the best model
with open("model.pkl", 'wb') as f:
    pickle.dump(best_model, f)

#with open("model.pkl", 'wb') as f:
 #   pickle.dump(model, f)
