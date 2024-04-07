import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #added standardscaler
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

#model = LogisticRegression().fit(X, y)
#model = LogisticRegression(max_iter=2000).fit(X, y)
model = LogisticRegression(penalty='l2', C=1.0, max_iter=2000).fit(X_scaled, y) #regularization with scaling


with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
