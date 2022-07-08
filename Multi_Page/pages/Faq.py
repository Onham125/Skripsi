import streamlit as st

st.header('FAQ :')

st.subheader('Q1 : How does this app work?')
st.write('A1 : First Users need to input parameter that we already provide in homepage after that users input parameter got normalize after that We Use XGBoost Method to  Predict Users input parameters has CKD or not ')

st.subheader('Q2 : What is CKD?')
st.write('A2 : Chronic Kidney Disease is a global public health problem with increased prevalence and incidence of renal failure, poor prognosis, and high cost. The prevalence of Chronic Kidney Disease  increases with the increase in the elderly and the development of diabetes and hypertension. Chronic Kidney Disease  has no signs or symptoms at first, but can gradually develop into renal failure. Kidney disease can be prevented and treated, and if detected early, it is more likely to be effective. ')

st.subheader('Q3 : What is XGBoost?')
st.write('A3 : XGBoost (Extreme Gradient Boosting) is a model proposed by Tianqi Chen and Carlos Guestrin (2016). This model has been optimized and improved in subsequent studies by many scientists. This model is a learning framework based on the Boosting Tree model. ')

st.subheader('Q4 : Does the Prediction Accuracy is Accurate?')
st.write('A4 :  According to our research, comparing 5 classification algorithms / methods such as  Naive Bayes, Decision Tree, Random Forest, k next Neighbor, XGBOOST  is the most accurate, and  XGBOOST has 10 times faster system execution speed. It has very good features such as It is the current solution on a single computer and can be scaled to billions of instances in a distributed or memory-constrained environment. Another advantage of the XGBoost algorithm is that parallel distributed computing accelerates learning. This speeds up model exploration. Another important point is that XGBoost will leverage off-core computing to enable data scientists to process hundreds of millions of samples on their desktops. XGBoost can combine these techniques to create an end-to-end system that scales larger datasets with minimal cluster resources   ')
