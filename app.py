import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("house_prices.csv")
    
    # Clean the 'Carpet Area' column
    df['Carpet Area'] = df['Carpet Area'].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)").astype(float)

    # Clean the 'Price (in rupees)' column
    df['Price (in rupees)'] = df['Price (in rupees)'].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)").astype(float)

    # Drop rows with missing or invalid data
    df = df.dropna(subset=['Carpet Area', 'Price (in rupees)'])
    df = df[df['Carpet Area'] > 0]
    df = df[df['Price (in rupees)'] > 0]

    return df


df = load_data()

# Prepare features
X = df[['Carpet Area']]
y = df['Price (in rupees)']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🏠 House Price Predictor")
st.markdown("This app predicts **house prices** based on carpet area using a Linear Regression model.")

carpet_input = st.number_input("Enter Carpet Area (in sqft):", min_value=0)

if carpet_input:
    prediction = model.predict(np.array([[carpet_input]]))[0]
    st.subheader(f"💰 Predicted Price: ₹{int(prediction):,}")

st.markdown("---")
st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error:** ₹{mae:,.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Plot
st.subheader("🔍 Visual Comparison")
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='Actual')
ax.plot(X_test, y_pred, color='red', label='Predicted')
ax.set_xlabel("Carpet Area (sqft)")
ax.set_ylabel("Price (₹)")
ax.legend()
st.pyplot(fig)
