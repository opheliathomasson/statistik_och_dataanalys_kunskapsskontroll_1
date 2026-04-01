import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st 

st.title(":red[Car price prediction]")
st.write("Fill in the car details to get an estimated price")

df = pd.read_csv("car_price_dataset.csv", sep=";", encoding="utf-8") # Load the car price dataset from a CSV file to prepare for training the linear regression model

X = df.drop("Price", axis=1)
y = df["Price"]

X = pd.get_dummies(X, drop_first=True, dtype=int) # Convert categorical variables into dummy/indicator variables to prepare the data for training the linear regression model

model = LinearRegression()
model.fit(X, y)

st.subheader("Fill out the car's information")
brand = st.selectbox("Brand", df["Brand"].unique())
model_name = st.selectbox("Model", df["Model"].unique())

year = st.number_input("Year", value=2015)

engine_size = st.number_input("Engine size", value=2.0)

fuel_type = st.selectbox("Fuel type", df["Fuel_Type"].unique())

transmission = st.selectbox("Transmission", df["Transmission"].unique())

mileage = st.number_input("Mileage", value=30000)

doors = st.number_input("Doors", value=2)

owner_count = st.number_input("Owner Count", value=1)

input_data = pd.DataFrame({"Brand": [brand], "Model": [model], "Year": [year], "Engine size": [engine_size], "Fuel Type": [fuel_type], 
                           "Transmission": [transmission], "Mileage": [mileage], "Doors": [doors], "Owner Count": [owner_count] }) # Create a DataFrame from the user input to prepare it for prediction using the trained linear regression model
input_data = pd.get_dummies(input_data, drop_first=True, dtype=int) # Convert the user input categorical variables into dummy/indicator variables to prepare it for prediction using the trained linear regression model
input_data = input_data.reindex(columns=X.columns, fill_value=0) # Ensure that the input data has the same columns as the training data (X) by reindexing and filling missing columns with zeros to prepare it for prediction using the trained linear regression model

if st.button("Predict price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted price: {int(prediction[0])} SEK") # Make a prediction using the trained linear regression model based on the user input and display the predicted price in Swedish Krona (SEK)
