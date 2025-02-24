import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns



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

# Evaluate the model performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Count plot for fraud vs non-fraud transactions
plt.figure(figsize=(6,4))
sns.countplot(x=df['is_fraud'], palette=['blue', 'red'])
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Transaction Type (0 = Legit, 1 = Fraud)')
plt.ylabel('Count')
plt.show()



