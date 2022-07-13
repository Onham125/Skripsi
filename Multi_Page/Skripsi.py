
import streamlit as st
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


st.sidebar.header('User Input Parameters')

Anemia =st.sidebar.radio('Anemia',  ('Yes','No'))
Appetite =st.sidebar.radio('Appetite', ('Good','Poor'))
Serum_Creatinine =st.sidebar.number_input('Serum_Creatinine',min_value=0.00)
Pedal_Edema=st.sidebar.radio('Pedal_Edema',  ('Yes','No'))
Sodium =st.sidebar.number_input('Sodium',min_value=0.00)
Red_Blood_Cells = st.sidebar.radio('Red_Blood_Cells',('Normal','Abnormal'))
Sugar = st.sidebar.number_input('Sugar',min_value=0.00)
Pus_Cell_Clumps = st.sidebar.radio('Pus_Cell_Clumps',('Present','NotPresent'))
Coronary_Artery_Disease = st.sidebar.radio('Coronary_Artery_Disease', ('Yes','No'))
Blood_Pressure = st.sidebar.number_input('Blood_Pressure',min_value=0.00)
Bacteria = st.sidebar.radio('Bacteria',('Present','NotPresent'))
White_Blood_Cell_Count =st.sidebar.number_input('White_Blood_Cell_Count',min_value=0.00)
Potassium =st.sidebar.number_input('Potassium',min_value=0.00)






#untuk user input parameter yang boolean
if Red_Blood_Cells == 'Normal':
    Red_Blood_Cells= 1
else :
    Red_Blood_Cells =0

if Pus_Cell_Clumps == 'Present':
    Pus_Cell_Clumps = 0
else : 
    Pus_Cell_Clumps = 1
if Bacteria == 'Present':
    Bacteria = 0
else :
    Bacteria = 1
if Coronary_Artery_Disease == 'Yes':
    Coronary_Artery_Disease = 0
else :
     Coronary_Artery_Disease = 1
if Appetite == 'Good' :
    Appetite = 1
else : 
    Appetite = 0
if Pedal_Edema == 'Yes':
    Pedal_Edema = 0
else :
    Pedal_Edema =1
if Anemia == 'Yes':
    Anemia = 0
else :
    Anemia =1

data = {
       'Anemia':Anemia,
       'Appetite': Appetite,
       'Serum_Creatinine':Serum_Creatinine,
       'Pedal_Edema' :Pedal_Edema,
       'Sodium': Sodium,
       'Red_Blood_Cells':Red_Blood_Cells,
       'Sugar': Sugar,
       'Pus_Cell_Clumps':Pus_Cell_Clumps,
       'Coronary_Artery_Disease':Coronary_Artery_Disease,
       'Blood_Pressure':Blood_Pressure,
       'Bacteria':Bacteria,
       'White_Blood_Cell_Count': White_Blood_Cell_Count,
       'Potassium': Potassium

    
            }

features = pd.DataFrame({'Label' :['Anemia','Appetite','Serum_Creatinine', 'Pedal_Edema', 'Sodium',
               'Red_Blood_Cells','Sugar','Pus_Cell_Clumps', 'Coronary_Artery_Disease', 
               'Blood_Pressure', 'Bacteria','White_Blood_Cell_Count','Potassium'] ,
    'Input User':[Anemia,Appetite,Serum_Creatinine,Pedal_Edema,Sodium,Red_Blood_Cells,Sugar,Pus_Cell_Clumps,Coronary_Artery_Disease,Blood_Pressure,Bacteria,White_Blood_Cell_Count,Potassium]})

df = features

data2 =pd.DataFrame(data,index=[0])

st.subheader('User Input parameters')
st.dataframe(data2)

dfasli = pd.read_csv('CKD_KNNFull1.csv')

#df1 = pd.read_csv('data chornic train 70persen.csv')
#df2 = pd.read_csv('data chornic test 30persen.csv')
dfKNNTrain = pd.read_csv('KNNtrain98XGBOOST.csv')
dfKNNTest = pd.read_csv('KNNtest98XGBOOST.csv')
#df_median70 = pd.read_csv('data chornic train 70persen handling missing value pakai median.csv')
#df_median30 = pd.read_csv('data chornic test 30persen handling missing value pakai median.csv');
dftest =dfasli.iloc[:,:-1]

## menggunakan concat untuk menambahkan user input parameter ke df yang missing value nya diisi dengan knnimputer dan setelah itu di normalize
test =pd.concat([dftest,data2],ignore_index=True)
test.reset_index()
df5 = pd.DataFrame(test,columns=dftest.columns)
scaler = MinMaxScaler(feature_range=(0, 1))
norm = scaler.fit_transform(df5)
norm = pd.DataFrame(norm, columns=df5.columns)
#norm
st.subheader('User Input Parameters (Normalize)')
testing = norm.iloc[-1:]
testing
#X_70 = df1.iloc[:,:-1] # Using all column except for the last column as X
#Y_70 = df1.iloc[:,-1] # Selecting the last column as Y

#X_30 = df2.iloc[:,:-1] #
#Y_30 = df2.iloc[:,-1] 

#Df_train = dfKNNTrain.iloc[:,:-1]
#Df_test = dfKNNTest.iloc[:,-1]

X_70 = dfKNNTrain.iloc[:,:-1]# Mengambil semua column kecuali label 
Y_70 = dfKNNTrain.iloc[:,-1]# mengambil hanya column label 
X_30 =dfKNNTest.iloc[:,:-1]# Mengambil semua column kecuali label 
Y_30 = dfKNNTest.iloc[:,-1]# mengambil hanya column label 

#X = df1.iloc[:,:-1] # Using all column except for the last column as X
#Y = df1.iloc[:,-1] # Selecting the last column as Y
#x_train,x_test ,y_train,y_test = train_test_split(X,Y,test_size=0.3)
#NB = GaussianNB()
#NBMed = GaussianNB()

#
#rf = RandomForestClassifier(max_depth=10)
#dc = DecisionTreeClassifier(min_samples_leaf=2,max_depth=10,min_samples_split=4)
#rf = dc.fit(X_70,Y_70)
#dc =rf.fit(X_70,Y_70)
#prediction = rf.predict(testing)
#predict2 =dc.predict(testing)

#KNN = KNeighborsClassifier(n_neighbors=7,
    #                       weights='uniform',
   #                        algorithm='auto',
  #                         leaf_size=30,
 #                          p=2,
 #                          metric='minkowski',metric_params=None,n_jobs=None)

xgb_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='', learning_rate=0.300000012, max_delta_step=0,
              max_depth=5, min_child_weight=1,
              monotone_constraints='()', n_estimators=100, n_jobs=8,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)
xgb_cl = xgb_cl.fit(X_70,Y_70)
xgbpred = xgb_cl.predict(testing)


#KNN = KNN.fit(X_70,Y_70)
#KNNpredict = KNN.predict(testing)


#NB =NB.fit(X_70,Y_70)
#NBPredict = NB.predict(testing)
#Median

#X_70med = df_median70.iloc[:,:-1]
#Y_70med =  df_median70.iloc[:,-1] # Select

#X_30med = df_median30.iloc[:,:-1] #
#Y_30med =  df_median30.iloc[:,-1] 

#rfmed = RandomForestClassifier(max_depth=10)
#dcmed = DecisionTreeClassifier(min_samples_leaf=2,max_depth=10,min_samples_split=4)
#dcmed =dcmed.fit(X_70med,Y_70med)
#rfmed =rfmed.fit(X_70med,Y_70med)
#rfmedpred = rfmed.predict(testing)
#dcmedpred =dcmed.predict(testing)

#KNNmed = KNeighborsClassifier(n_neighbors=7,
 #                          weights='uniform',
  #                         algorithm='auto',
   #                        leaf_size=30,
    #                       p=2,
     #                      metric='minkowski',metric_params=None,n_jobs=None)

#xgb_clMed =xgb.XGBClassifier(objective='binary:logistic')
#xgb_clMed =xgb_clMed.fit(X_70med,Y_70med)
#xgbMedPred = xgb_clMed.predict(testing)

#NBmed = GaussianNB()
#NBmed =NBmed.fit(X_70med,Y_70med)
#NBPredictmed = NBmed.predict(testing)

#KNNmed =KNNmed.fit(X_70med,Y_70med)
#KNNmedpredict = KNNmed.predict(testing)




##Proses
#st.subheader('Class labels')
#st.write(pd.DataFrame({     'Label': ['CKD','Not CKD'],}))
#st.header('Prediction Using Mean ')
#st.subheader('Prediction Random Forest')
#PredCKDRF = rf.predict_proba(testing)[:,0]
#prednonCKDRF = rf.predict_proba(testing)[:,1]
#if PredCKDRF>prednonCKDRF:
  #  hasilrf = str(int(PredCKDRF[0]*100))+'% Terkena CKD'
   # hasilrfintckd = int(PredCKDRF[0]*100)
    #rfintnonckd=0

#if  prednonCKDRF>PredCKDRF:
    #hasilrf = str(int(prednonCKDRF[0]*100))+'% Tidak Terkena CKD'
    #rfintnonckd = int(prednonCKDRF[0]*100)
    #hasilrfintckd =0
#st.write(hasilrf)
#st.write((rf.predict_proba(testing)))

#st.subheader('Prediction Decision Tree')
#predCKDDC = dc.predict_proba(testing)[:,0]
#prednonCKDdc = dc.predict_proba(testing)[:,1]
#if predCKDDC>prednonCKDdc:
#    hasildc = str(int(predCKDDC[0]*100))+'% Terkena CKD'
#    hasildcintckd = int(predCKDDC[0]*100)
#    hasildcint=0
#if prednonCKDdc>predCKDDC:
 #   hasildc = str(int(prednonCKDdc[0]*100))+'% Tidak Terkena CKD'
 #   hasildcint = int(prednonCKDdc[0]*100)
 #   hasildcintckd =0
#st.write(hasildc)
#st.write((dc.predict_proba(testing)))

#st.subheader('Prediction KNN')
#predCKDKNN = KNN.predict_proba(testing)[:,0]
#prednonCKDKNN = KNN.predict_proba(testing)[:,1]
#if predCKDKNN>prednonCKDKNN:
 #   hasilknn = str(int(predCKDKNN[0]*100))+'% Terkena CKD'
  #  hasilknnintckd = int(predCKDKNN[0]*100)
   # hasilknnint=0
#if prednonCKDKNN>predCKDKNN:
 #   hasilknn = str(int(prednonCKDKNN[0]*100))+'% Tidak Terkena CKD'
#hasilknnint = int(prednonCKDKNN[0]*100)
 #   hasilknnintckd =0
#st.write(hasilknn)
#st.write((KNN.predict_proba(testing)))



#st.subheader('Prediction Naive Bayes')
#predCKDNB = NB.predict_proba(testing)[:,0]
#prednonCKDNB = NB.predict_proba(testing)[:,1]
#if predCKDNB>prednonCKDNB:
    #hasilnb = str(int(predCKDNB[0]*100))+'% Terkena CKD'
   # hasilnbintckd = int(predCKDNB[0]*100)
   # hasilnbint =0
#if  prednonCKDNB>predCKDNB:
    #hasilnb = str(int(prednonCKDNB[0]*100))+'% Tidak Terkena CKD'
   # hasilnbint = int(prednonCKDNB[0]*100)
  #  hasilnbintckd =0
#st.write(hasilnb)



#st.subheader('Prediction XGBoost')


## Mengambil hasil prediksi dari predict_proba
predCKDXGB = xgb_cl.predict_proba(testing)[:,0]
prednonCKDXGB = xgb_cl.predict_proba(testing)[:,1]
if predCKDXGB>prednonCKDXGB:
    hasilXGB = 'CKD'
   # hasilXGBintckd = int(predCKDXGB[0]*100)
    #hasilXGBint =0
if  prednonCKDXGB> predCKDXGB:
    hasilXGB = 'Not CKD'
    #hasilXGBint = int(prednonCKDXGB[0]*100)
    #hasilXGBintckd=0

#st.write(hasilXGB)

#st.write(xgb_cl.predict_proba(testing))
#total_mean_ckd = int(hasilrfintckd+hasilXGBintckd+hasilknnintckd+hasildcintckd)/4
#total_mean_ckd
#total_mean_nonckd = int(rfintnonckd+hasilXGBint+hasilknnint+hasildcint)/4
#total_mean_nonckd
#if(total_mean_ckd>total_mean_nonckd):
  #  hasil_mean = 'CKD'
#else:
  #  hasil_mean = 'Not CKD'

st.header('Predictions Result:')
st.subheader(hasilXGB)


#st.header('Prediction Using Median ')
#st.subheader('Prediction Random Forest')
#st.write((rfmed.predict_proba(testing)))
#st.subheader('Prediction Decision Tree')
#st.write((dcmed.predict_proba(testing)))
#st.subheader('Prediction KNN')
#st.write((KNNmed.predict_proba(testing)))
#st.subheader('Prediction Naive Bayes')
#st.write((NBmed.predict_proba(testing)))
#st.subheader('Prediction XGBoost')
#st.write(xgb_clMed.predict_proba(testing))
# clf = RandomForestClassifier()
# clf.fit(X, Y)

# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)

#
# st.write(iris.target_names)

# st.subheader('Prediction')
# st.write(iris.target_names[prediction])
# #st.write(prediction)

# st.subheader('Prediction Probability')
# st.write(prediction_proba)




