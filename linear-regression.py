import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Titanic Survival Predictor", icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger's details below to see their chance of survival.")

# --- STEP 1: LOAD THE MODEL ---
# This looks for the 'titanic_model.pkl' file you uploaded to GitHub
model_path = 'titanic_model.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    st.error(f"❌ Error: '{model_path}' not found! Please upload the model file to GitHub.")
    st.stop()

# --- STEP 2: USER INPUTS (Sidebar) ---
st.sidebar.header("Passenger Profile")

pclass = st.sidebar.selectbox("Ticket Class (1=High, 3=Low)", [1, 2, 3])
sex = st.sidebar.radio("Gender", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings or Spouses aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents or Children aboard", 0, 10, 0)
fare = st.sidebar.number_input("Ticket Fare ($)", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# --- STEP 3: PREPROCESS INPUTS ---
# Convert 'sex' to 0 or 1
sex_val = 1 if sex == "female" else 0

# Convert 'embarked' to the dummy variables we used in training (Embarked_Q, Embarked_S)
emb_q = 1 if "Q" in embarked else 0
emb_s = 1 if "S" in embarked else 0

# Put all 8 features in a list (The order must match your training data!)
# [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]
features = np.array([[pclass, sex_val, age, sibsp, parch, fare, emb_q, emb_s]])

# --- STEP 4: PREDICTION ---
if st.button("Predict Survival Status"):
    # Get the 0 or 1 prediction
    prediction = model.predict(features)
    
    # Get the probability (percentage)
    probability = model.predict_proba(features)[0][1]
    
    st.divider()
    
    if prediction[0] == 1:
        st.balloons()
        st.success(f"### Result: Survived! 🎉")
        st.write(f"The model is **{probability:.2%}** confident this passenger would survive.")
    else:
        st.error(f"### Result: Did Not Survive 😔")
        st.write(f"The model is **{(1-probability):.2%}** confident this passenger would not survive.")

# --- STEP 5: INTERPRETATION (Optional UI) ---
with st.expander("How does this work?"):
    st.write("""
    This app uses a **Logistic Regression** model. It looks at factors like 
    class and gender—which were historically significant during the Titanic 
    disaster—to calculate a probability of survival.
    """)
