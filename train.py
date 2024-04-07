import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier #added decision tree
from sklearn.preprocessing import StandardScaler #added standardscaler
from sklearn.model_selection import GridSearchCV #added grid search
from sklearn.ensemble import RandomForestClassifier #added randomforest
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

#df = pd.read_csv("data/train.csv")
#X = df.drop(columns=['Disease']).to_numpy()
#y = df['Disease']

# Encoding the target variable
#label_encoder = LabelEncoder()
#y = label_encoder.fit_transform(y)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM with Hyperparameter Tuning
#param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
#svm_model = SVC()
#grid_search = GridSearchCV(svm_model, param_grid)
#grid_search.fit(X_scaled, y) 

# Decision Tree with Hyperparameter Tuning
#param_grid = {
 #   'max_depth': [2, 3, 4, 5],  
  #  'min_samples_split': [2, 5, 10, 15],
 #   'ccp_alpha': [0.01, 0.1, 0.5, 1, 5, 10] 
#}
#param_grid = {'max_depth': [3, 5, 8],  
#              'min_samples_split': [2, 5, 10]}

#dt_model = DecisionTreeClassifier()
#grid_search = GridSearchCV(dt_model, param_grid)
#grid_search.fit(X_scaled, y)  # Missing values will be handled by the decision tree

# Hyperparameter Tuning with GridSearchCV
#param_grid = {'C': [5, 3, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]}  # Values of 'C' to try
#grid_search = GridSearchCV(LogisticRegression(penalty='l2', max_iter=5000), param_grid)
#grid_search.fit(X_scaled, y)

# Hyperparameter Tuning for logistic regression
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 30, 50, 68, 100]}  # Adjust the range of 'C' as needed
grid_search_lr = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000), param_grid_lr)
grid_search_lr.fit(X_scaled, y)

# Get the best Logistic Regression model
best_lr_model = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Logistic Regression model:", best_lr_model)

#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 30, 50, 68, 100]}  # Values of 'C' to try
#grid_search = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000), param_grid)
#grid_search.fit(X_scaled, y)

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(X_scaled, y)

# Random Forest with Hyperparameter Tuning
#param_grid = {
 #   'n_estimators': [100, 200, 300],
  #  'max_depth': [5, 10, 15],
   # 'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4]
#}

#rf_model = RandomForestClassifier(random_state=42)
#grid_search = GridSearchCV(rf_model, param_grid, cv=5)
#grid_search.fit(X_scaled, y)

# Get the best Logistic Regression model
best_lr_model = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Logistic Regression model:", best_lr_model)

# Get the best model
#print("Best parameters for Random Forest:", grid_search_rf.best_params_)
#print("Best Random Forest model:", best_rf_model)

#model = LogisticRegression().fit(X, y)
#model = LogisticRegression(max_iter=2000).fit(X, y)
#model = LogisticRegression(penalty='l0', C=1.0, max_iter=2000).fit(X_scaled, y) #regularization with scaling

# Save the best model (either Logistic Regression or Random Forest based on performance)
if grid_search_lr.best_score_ > grid_search_rf.best_score_:
    best_model = best_lr_model
else:
    best_model = best_rf_model

# Save the best model
with open("model.pkl", 'wb') as f:
    pickle.dump(best_model, f)

#with open("model.pkl", 'wb') as f:
 #   pickle.dump(model, f)
