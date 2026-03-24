import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("🏡 House Price Prediction")

# Upload file
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("### Data Preview")
    st.write(df.head())

    # Select features
    X = df[['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'AGE']]
    y = df['MEDV']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    st.write("### Model Performance")
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R2 Score:", r2_score(y_test, y_pred))

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    # User input
    st.write("### Predict Price")

    rm = st.number_input("Rooms", 0.0)
    lstat = st.number_input("LSTAT", 0.0)
    ptratio = st.number_input("PTRATIO", 0.0)
    indus = st.number_input("INDUS", 0.0)
    nox = st.number_input("NOX", 0.0)
    age = st.number_input("AGE", 0.0)

    if st.button("Predict"):
        data = np.array([[rm, lstat, ptratio, indus, nox, age]])
        result = model.predict(data)
        st.success(f"Predicted Price: {result[0]:.2f}")
