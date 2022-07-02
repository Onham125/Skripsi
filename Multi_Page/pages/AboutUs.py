from msilib.schema import Feature
from PIL import Image
import streamlit as st


st.header('About Us :')
st.markdown('We are Bina Nusantara College Student that aware of the Chronic Kidney Diseases and we also realize that in this day and age where technology is developing rapidly and technology has also been widely used for various types of science where technology can help in our daily activities. Technology is also helping in the healthcare sector where healthcare workers are greatly aided by the technology that helps them, but most people do not understand or lack knowledge about the symptoms of the disease chronic kidney disease.')
image = Image.open('pages\gambar.jpeg')
st.image(image, caption='Kidney')




st.subheader('CKD: ')
st.markdown('   Chronic Kidney Disease is a global public health problem with increased prevalence and incidence of renal failure, poor prognosis, and high cost. The prevalence of Chronic Kidney Disease  increases with the increase in the elderly and the development of diabetes and hypertension. Chronic Kidney Disease  has no signs or symptoms at first, but can gradually develop into renal failure. Kidney disease can be prevented and treated, and if detected early, it is more likely to be effective.')
st.subheader('Building Our Prediction App')
st.markdown('We use the UCI dataset to classify users input parameters that we provide on the Home page using XGBoost method/algorithm to predict whether the patient has CKD or not ')

st.write('The Dataset : [Link](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease)')