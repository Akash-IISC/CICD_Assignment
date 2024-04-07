import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier #added decision tree
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


# Hyperparameter Tuning for Logistic Regression
param_grid_lr = {'C': [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.6, 0.065, 0.07, 0.075]}
grid_search_lr = GridSearchCV(LogisticRegression(penalty='l2', max_iter=50000), param_grid_lr)
grid_search_lr.fit(X_scaled, y)

# Get the best Logistic Regression model
best_model_lr = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Logistic Regression model:", best_model_lr)


#model = LogisticRegression().fit(X, y)
#model = LogisticRegression(max_iter=2000).fit(X, y)
#model = LogisticRegression(penalty='l2', C=1.0, max_iter=2000).fit(X_scaled, y) #regularization with scaling

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

# Create a Voting Classifier with Logistic Regression and Random Forest
voting_classifier = VotingClassifier(estimators=[('lr', best_model_lr), ('rf', best_rf_model)], voting='soft')

# Train the Voting Classifier
voting_classifier.fit(X_scaled, y)

# Save the best Voting Classifier model
with open("model.pkl", 'wb') as f:
    pickle.dump(voting_classifier, f)

# Save the best model
#with open("model.pkl", 'wb') as f:
 #   pickle.dump(best_model, f)

#with open("model.pkl", 'wb') as f:
 #   pickle.dump(model, f)
