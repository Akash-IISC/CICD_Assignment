import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle

#Load the data
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()


# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter Tuning for Logistic Regression
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Adjust the range of 'C' as needed
grid_search_lr = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000), param_grid_lr)
grid_search_lr.fit(X_scaled, y)

# Get the best Logistic Regression model
best_lr_model = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Logistic Regression model:", best_lr_model)

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(X_scaled, y)

# Get the best Random Forest model
best_rf_model = grid_search_rf.best_estimator_
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Random Forest model:", best_rf_model)

# Save the best model (either Logistic Regression or Random Forest based on performance)
if grid_search_lr.best_score_ > grid_search_rf.best_score_:
    best_model = best_lr_model
else:
    best_model = best_rf_model

with open("model.pkl", 'wb') as f:
    pickle.dump(best_model, f)
