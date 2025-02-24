import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score


# Scikit-learn modules for splitting, scaling, encoding, and model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# SMOTE for handling imbalanced classes
from imblearn.over_sampling import SMOTE

# Load the dataset (change 'creditcard_transactions.csv' to your file path)
df = pd.read_csv('fraudTest.csv')

# Convert the transaction date/time column to datetime object
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Create a new feature: the hour of the transaction
df['trans_hour'] = df['trans_date_trans_time'].dt.hour

# Initialize a label encoder
le = LabelEncoder()

# Encode the merchant, category, and gender columns
df['merchant_enc'] = le.fit_transform(df['merchant'])
df['category_enc'] = le.fit_transform(df['category'])
df['gender_enc'] = le.fit_transform(df['gender'])


# Define the features and the target variable
features = ['amt', 'trans_hour', 'merchant_enc', 'category_enc', 'gender_enc']
target = 'is_fraud'

# Create feature matrix X and target vector y
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train[['amt', 'trans_hour']] = scaler.fit_transform(X_train[['amt', 'trans_hour']])
X_test[['amt', 'trans_hour']] = scaler.transform(X_test[['amt', 'trans_hour']])


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained! Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, "fraud_detection_model.pkl")
print("💾 Model saved as fraud_detection_model.pkl")