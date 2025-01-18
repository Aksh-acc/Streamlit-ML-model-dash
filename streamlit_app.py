import streamlit as st
import pandas as pd
st.title('Machine Learning prediction dash')
st.info('This is a machine learing model app where you can get the insights from the data and get the predictions along with')
df = pd.read_csv('https://raw.githubusercontent.com/Aksh-acc/Streamlit-ML-model-dash/refs/heads/master/Iris.csv')
with st.expander("data"):
  st.write('**raw data**')
  st.write(df)
  st.write('**X value**')
  X = df.drop('Species' ,axis =1)
  X
  st.write('**Y value**')
  y = df.Species
  y

with st.sidebar:
  sepal_length_in_cm 	 =st.slider('sepal_length_in_cm' ,4.30 ,7.90 ,5.80)
  sepal_width_in_cm 	 =st.slider('sepal_width_in_cm' ,2.00 ,4.40 ,3.00)
  petal_length_in_cm 	 =st.slider('petal_length_in_cm' ,1.00 ,6.90,4.35)
  petal_width_in_cm 	 =st.slider('petal_width_in_cm' ,0.10 ,2.50,1.30)
  
  data ={
    'sepal_length': sepal_length_in_cm 	,
    'sepal_width' :sepal_width_in_cm ,
    'petal_length' : petal_length_in_cm ,	
   ' petal_width':petal_width_in_cm
  }
  input_df =pd.DataFrame(data ,index =[0])
  input_df
    
from sklearn import model_selection 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC

models = [] 
models.append(('LR', LogisticRegression())) 
models.append (('IDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier()))
models.append (("CART", DecisionTreeClassifier()))
models.append (('NB', GaussianNB()))
models.append (('SVM', SVC())) 

