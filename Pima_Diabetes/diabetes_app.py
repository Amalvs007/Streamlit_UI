import streamlit as st 
import pickle
import pandas as pd
import joblib

# Load the model file
model = joblib.load("model.pkl")
scale = joblib.load("scale.pkl")

st.title("Diabetes Detection Using AI...!")

st.header("The Data Frame used")
df = pd.read_csv("diabetes.csv")
st.dataframe(df.head())

col1,col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies')
    glucose = st.number_input('Glucose')
    bloodPressure = st.number_input('Blood Pressure')
    skinThickness = st.number_input('Skin Thickness')
with col2:
    insulin = st.number_input('Insulin')
    bmi = st.number_input('BMI')
    dpf = st.number_input('DPF')
    age = st.number_input('Age')

if st.button('Predict Diabetes'):
    # st.write('The Pregnancy count is ', int(pregnancies))
    # st.write('The Glucose count is ', int(glucose))
    # st.write('The Blood Pressure count is ', int(bloodPressure))
    # st.write('The Skin Thickness count is ', int(skinThickness))
    # st.write('The Insulin count is ', int(insulin))
    # st.write('The BMI count is ', int(bmi))
    # st.write('The DPF count is ', int(dpf))
    # st.write('The Age count is ', int(age))

    rowDF = pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])])
    rowDF_new = pd.DataFrame(scale.transform(rowDF))

    st.subheader("The Independent variables given by the user")
    st.table(rowDF_new)

    # Model Prediction
    prediction = model.predict_proba(rowDF_new)
    st.subheader("The Predicted Probabilities")
    st.write(prediction)

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        #print(f"The Round Value : {valPred*100}%")
        st.warning(f'You have a chance of having Diabetes. \n\nProbability of you being a Diabetic is {valPred*100:.2f}%. \n\nAdvice : Exercise Regularly', icon="⚠️")
    else:
        valPred = round(prediction[0][0],3)
        #print(f"The Round Value : {valPred*100}%")
        st.success(f'Congratulations!!!, You are in a SAFE ZONE. \n\nProbability of you being a NON-Diabetic is {valPred*100:.2f}%. \n\nAdvice : Exercise Regularly and Maintain like this..!', icon="✅")
