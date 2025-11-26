import streamlit as st
import pandas as pd
import joblib

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")

model = joblib.load(MODEL_PATH)


st.title("🛒 Customer Purchase Prediction")

st.write("Nhập thông tin khách hàng:")

inputs = {}

inputs["Administrative"] = st.number_input("Administrative", 0, 20)
inputs["Administrative_Duration"] = st.number_input("Administrative Duration", 0.0)
inputs["Informational"] = st.number_input("Informational", 0, 20)
inputs["Informational_Duration"] = st.number_input("Informational Duration", 0.0)
inputs["ProductRelated"] = st.number_input("ProductRelated", 0, 100)
inputs["ProductRelated_Duration"] = st.number_input("ProductRelated Duration", 0.0)
inputs["BounceRates"] = st.number_input("BounceRates", 0.0, 1.0)
inputs["ExitRates"] = st.number_input("ExitRates", 0.0, 1.0)
inputs["PageValues"] = st.number_input("PageValues", 0.0)
inputs["SpecialDay"] = st.selectbox("SpecialDay", [0.0, 0.2, 0.4, 0.6, 0.8])
inputs["Month"] = st.selectbox("Month", 
    ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
inputs["OperatingSystems"] = st.number_input("OperatingSystems", 1, 10)
inputs["Browser"] = st.number_input("Browser", 1, 10)
inputs["Region"] = st.number_input("Region", 1, 10)
inputs["TrafficType"] = st.number_input("TrafficType", 1, 20)
inputs["VisitorType"] = st.selectbox("VisitorType", ["New_Visitor","Returning_Visitor","Other"])
inputs["Weekend"] = st.selectbox("Weekend", [True, False])

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)[0]
    st.success("Kết quả: **CÓ MUA**" if pred == 1 else "KHÔNG MUA")
