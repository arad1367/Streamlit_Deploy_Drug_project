# -*- coding: utf-8 -*-
"""
@author: Pejman Ebrahimi
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from PIL import Image

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# Define the function
def predict_drug_type(Age, Sex, BP, Cholesterol, Na_to_K):
    """
    Predict the drug type using a pre-trained machine learning pipeline.

    Args:
        Age: Patient's age.
        Sex: Patient's sex.
        BP: Patient's blood pressure.
        Cholesterol: Patient's cholesterol level.
        Na_to_K: Sodium-to-Potassium ratio.

    Returns:
        The predicted outcome of drug type.
    """
    # Make sure to replace 'pipeline' with the actual name of your trained pipeline
    return classifier.predict(pd.DataFrame([[Age, Sex, BP, Cholesterol, Na_to_K]],
                                         columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']))



def main():
    st.title("Drug Type")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Drug type ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age = st.text_input("Age","Type Here")
    Sex = st.text_input("Sex","Type Here")
    BP = st.text_input("BP","Type Here")
    Cholesterol = st.text_input("Cholesterol","Type Here")
    Na_to_K = st.text_input("Na_to_K","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_drug_type(Age, Sex, BP, Cholesterol, Na_to_K)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn to predict Drug type with a ML model in easy way")
        st.text("Built with Streamlit")
        st.text("@author: Pejman Ebrahimi")
        st.text("Made with love")

if __name__=='__main__':
    main()