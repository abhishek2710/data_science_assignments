import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'titanic_model.pkl')

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    st.error("The file 'titanic_model.pkl' was not found in the GitHub folder!")
    
st.set_page_config(page_title="Titanic Predictor")
st.title("🚢 Titanic Survival Predictor")

# Sidebar inputs
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Fare", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Prepare input data (Must match the 8 features used in training)
sex_val = 1 if sex == 'female' else 0
emb_q = 1 if embarked == 'Q' else 0
emb_s = 1 if embarked == 'S' else 0

input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, emb_q, emb_s]])

if st.button("Calculate Survival Probability"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction[0] == 1:
        st.success(f"High probability of survival! ({probability:.2%})")
    else:
        st.error(f"Low probability of survival. ({probability:.2%})")
