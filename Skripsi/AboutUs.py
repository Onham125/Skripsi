from Multi_Page.Skripsischema import Feature
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


st.header('About Us :')

image = Image.open('image.jpg')
st.image(image, caption='Kidney')