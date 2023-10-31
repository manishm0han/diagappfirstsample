import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Med Diag Web App ⚕️')
st.subheader('Does the person have diabetes?')
df=pd.read_csv('diabetes.csv')
if st.sidebar.checkbox('View Data',False):
    st.write(df)
if st.sidebar.checkbox('View Distributuons',False):
    df.hist()
    plt.tight_layout()
    st.pyplot()
    
# load pickled model
model=open('rfc.pickle','rb')
clf=pickle.load(model)
model.close()

# get front end user input
pregs=st.number_input('Pregnancies',0,20,0)
plas=st.slider('Glucose',40,200,0)
pres=st.slider('BloodPressure',20,150,20)
skin=st.slider('SkinThickness',7,99,7)
ins=st.slider('Insulin',14,850,14)
bmi=st.slider('BMI',18,70,18)
diaped=st.slider('DiabetesPedigreeFunction',0.05,2.50,0.05)
age=st.slider('Age',21,90,21)

#Get model input
input_data=[[pregs,plas,pres,skin,ins,bmi,diaped,age]]

#predict and print
prediction=clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction==0:
        st.subheader('Non diabetic')
    else:
        st.subheader('Diabetic')
