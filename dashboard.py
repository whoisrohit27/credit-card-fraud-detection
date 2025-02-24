import streamlit as st
import pandas as pd
import joblib  # To load the trained model
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("fraudTrain.csv")  # Replace with actual dataset path
    return df

df = load_data()

# Load Trained Model
model = joblib.load("fraud_detection_model.pkl")  # Replace with your model path

# Sidebar Filters
st.sidebar.header("Filter Transactions")
category = st.sidebar.multiselect("Select Category", df["category"].unique())
state = st.sidebar.multiselect("Select State", df["state"].unique())

filtered_df = df.copy()
if category:
    filtered_df = filtered_df[filtered_df["category"].isin(category)]
if state:
    filtered_df = filtered_df[filtered_df["state"].isin(state)]

# Dashboard Title
st.title("💳 Credit Card Fraud Detection Dashboard")

# Fraud Distribution
st.subheader("Fraud vs Non-Fraud Transactions")
fig, ax = plt.subplots()
sns.countplot(x="is_fraud", data=df, palette=["blue", "red"], ax=ax)
st.pyplot(fig)

# Fraud Transactions by Category
st.subheader("Fraudulent Transactions by Category")
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(y="category", hue="is_fraud", data=df, palette=["blue", "red"], ax=ax)
st.pyplot(fig)

# Fraud by Time of Day
st.subheader("Fraudulent Transactions by Hour")
df["trans_hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
fig, ax = plt.subplots()
sns.histplot(df[df["is_fraud"] == 1]["trans_hour"], bins=24, color='red', kde=True, label="Fraud")
sns.histplot(df[df["is_fraud"] == 0]["trans_hour"], bins=24, color='blue', kde=True, label="Legit")
plt.legend()
st.pyplot(fig)

# Predict Fraud for New Transaction
st.subheader("🔍 Predict Fraud for a New Transaction")
amt = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=10000.0, value=50.0)
merchant = st.selectbox("Merchant Category", df["category"].unique())
state = st.selectbox("Transaction State", df["state"].unique())
hour = st.slider("Transaction Hour", 0, 23, 12)

# Convert Inputs to Model Format
new_data = pd.DataFrame([[amt, merchant, state, hour]], columns=["amt", "category", "state", "trans_hour"])

# Predict Button
if st.button("Check Fraud Probability"):
    prediction = model.predict(new_data)[0]
    fraud_prob = model.predict_proba(new_data)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 This transaction is **likely FRAUD** with {fraud_prob:.2%} probability!")
    else:
        st.success(f"✅ This transaction is **legit** with {fraud_prob:.2%} probability.")

# Show Filtered Data
st.subheader("📊 Filtered Transactions")
st.write(filtered_df.head(50))  # Display first 50 filtered rows

