import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# 1. Setup Page
st.set_page_config(page_title="Titanic Survival Predictor")
st.title("🚢 Titanic Survival Predictor")

# 2. Find and Load the Model
# This code ensures it works on both your computer and GitHub
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'titanic_model.pkl')

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    st.error(f"⚠️ Could not find '{model_path}'. Make sure you uploaded it to GitHub!")
    st.stop()

# 3. Create the User Interface
st.write("Enter passenger details to predict if they would have survived:")

# Inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class", [1, 2, 3], help="1 = Upper, 2 = Middle, 3 = Lower")
    age = st.slider("Age", 0, 100, 25)
    sex = st.radio("Gender", ["male", "female"])

with col2:
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children aboard", 0, 10, 0)
    fare = st.number_input("Fare Paid ($)", 0.0, 500.0, 32.0)

# 4. Prepare Data for Prediction
# Convert 'female' to 1, 'male' to 0
sex_val = 1 if sex == "female" else 0

# We assume your model was trained on these 6 features in this order:
# [Pclass, Sex, Age, SibSp, Parch, Fare]
input_features = np.array([[pclass, sex_val, age, sibsp, parch, fare]])

# 5. Make Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)[0][1]
    
    st.markdown("---")
    if prediction[0] == 1:
        st.balloons()
        st.success(f"### Result: Likely Survived! 🎉")
        st.write(f"Confidence: **{probability:.2%}**")
    else:
        st.error(f"### Result: Likely Did Not Survive 😔")
        st.write(f"Confidence: **{(1-probability):.2%}**")
