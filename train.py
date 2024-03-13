# Importing necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
# Importing data
df = pd.read_csv('Crop_recommendation.csv')
#N,P,K,temperature,humidity,ph,rainfall,label
# Extracting features and target variable
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choosing a machine learning algorithm and training the model
model = LogisticRegression(max_iter=200)  # Increase max_iter
model.fit(X_train_scaled, y_train)

# Making predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
joblib.dump(model, 'logistic_regression_model.pkl')
