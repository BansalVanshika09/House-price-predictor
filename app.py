import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("house_prices_small.csv")
    
    # Clean the 'Carpet Area' column
    df['Carpet Area'] = df['Carpet Area'].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)").astype(float)

    # Clean the 'Price (in rupees)' column
    df['Price (in rupees)'] = df['Price (in rupees)'].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)").astype(float)

    # Drop rows with missing or invalid data
    df = df.dropna(subset=['Carpet Area', 'Price (in rupees)', 'Location'])
    df = df[df['Carpet Area'] > 0]
    df = df[df['Price (in rupees)'] > 0]

    return df

df = load_data()

# Define valid cities (from the dataset)
valid_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Jaipur', 'Rural']

# Prepare features
X = df[['Carpet Area', 'Location']]
y = df['Price (in rupees)']

# Preprocessing: One-hot encode Location, pass Carpet Area as-is
preprocessor = ColumnTransformer(
    transformers=[
        ('location', OneHotEncoder(categories=[valid_cities], handle_unknown='ignore'), ['Location']),
        ('carpet', 'passthrough', ['Carpet Area'])
    ])

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🏠 House Price Predictor")
st.markdown("This app predicts **house prices** based on carpet area and location using a Linear Regression model.")

# Input fields
carpet_input = st.number_input("Enter Carpet Area (in sqft):", min_value=0.0, step=1.0)
location_input = st.selectbox("Select Location:", valid_cities)

if carpet_input and location_input:
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([[carpet_input, location_input]], columns=['Carpet Area', 'Location'])
    
    # Predict price
    prediction = model.predict(input_data)[0]
    st.subheader(f"💰 Predicted Price: ₹{int(prediction):,}")

st.markdown("---")
st.subheader("📊 Model Performance")
st.write(f"**Mean Absolute Error:** ₹{mae:,.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Plot
st.subheader("🔍 Visual Comparison")
fig, ax = plt.subplots()
# Scatter plot for actual vs predicted (using Carpet Area as x-axis)
scatter = ax.scatter(X_test['Carpet Area'], y_test, c=X_test['Location'].astype('category').cat.codes, 
                     cmap='viridis', label='Actual', alpha=0.5)
ax.scatter(X_test['Carpet Area'], y_pred, color='red', label='Predicted', alpha=0.5)
ax.set_xlabel("Carpet Area (sqft)")
ax.set_ylabel("Price (₹)")
ax.legend()
# Add a colorbar for Location
plt.colorbar(scatter, label='Location (encoded)')
st.pyplot(fig)
