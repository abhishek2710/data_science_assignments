import streamlit as st
import pandas as pd
import pickle

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to see if they would have survived.")

# User Inputs
pclass = st.selectbox("Travel Class (1=Best)", [1, 2, 3])
sex = st.radio("Gender", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
fare = st.number_input("Ticket Fare", value=32.0)

# Convert inputs to match model features
sex_val = 1 if sex == "female" else 0

if st.button("Predict Survival"):
    # 1. Load the "Brain"
    with open('Titanic_train.csv', 'rb') as f:
        model = pickle.load(f)
    
    # 2. Package the data (adding 0s for missing features like SibSp/Parch)
    features = np.array([[pclass, sex_val, age, 0, 0, fare, 0, 1]])
    
    # 3. Get the answer
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
    if prediction[0] == 1:
        st.success(f"They likely survived! (Probability: {probability:.2%})")
    else:
        st.error(f"They likely did not survive. (Probability: {probability:.2%})")
