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
# with st.expander("Data Visualization"):
#     st.altair_chart(
#     data = df,
#     x=X,
#     y=y,
#     color='Species')
  
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
st.write('**Input data Values**')
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

X_train ,X_test ,y_train ,y_test  =model_selection.train_test_split(X ,y ,test_size =0.2 ,random_state =6)
names =[]
predictions =[]
model=[]
for name , model in models :
  kfold = model_selection.KFold(n_splits = 10  )
  cv_result = model_selection.cross_val_score(model  , X_train , Y_train , cv = kfold , scoring = scoring)
  results.append(cv_result)
  names.append(name)
  predictions.append("%s: %f (%f) " %(name , cv_result.mean() , cv_result.std()))
# print("names:" ,names) 
# print(msg)
parsed_data = []
for item in predictions:
    model, values = item.split(":")
    accuracy = float(values.split()[0])
    parsed_data.append((model.strip(), accuracy))

# Sort the data in descending order based on accuracy
sorted_data = sorted(parsed_data, key=lambda x: x[1], reverse=True)
first_model, first_val = sorted_data[0]
final_model =None
for model_name ,model in models:
    if model_name == first_model:
        final_model =model
        break
# print(final_model)
final_model.fit(X ,y)
prediction = final_model.predict(input_df)
prediction

