import streamlit as st
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Title and Info
st.title('🤖 Machine Learning Prediction Dashboard')
st.info('This is a machine learning model app where you can get insights from the data and make predictions.')
#upload your own custom dataset
# feature not working according to current programme will work in the future for this

# upload_file =st.file_uploader("Choose a csv file")
# df = pd.read_csv(upload_file) 


# Load and display  default dataset
df = pd.read_csv('https://raw.githubusercontent.com/Aksh-acc/Streamlit-ML-model-dash/refs/heads/master/Iris.csv')
with st.expander("Data"):
    st.write('**Raw Data**')
    st.write(df)
    st.write('**X Values**')
    X = df.drop(['Species', 'Id'], axis=1)
    st.write(X)
    st.write('**Y Values**')
    y = df['Species']
    st.write(y)

# Sidebar sliders for input
with st.sidebar:
    sepal_length_in_cm = st.slider('SepalLengthCm', 4.3, 7.9, 5.8)
    sepal_width_in_cm = st.slider('SepalWidthCm', 2.0, 4.4, 3.0)
    petal_length_in_cm = st.slider('PetalLengthCm', 1.0, 6.9, 4.35)
    petal_width_in_cm = st.slider('PetalWidthCm', 0.1, 2.5, 1.3)

    data = {
        'SepalLengthCm': sepal_length_in_cm,
        'SepalWidthCm': sepal_width_in_cm,
        'PetalLengthCm': petal_length_in_cm,
        'PetalWidthCm': petal_width_in_cm
    }
    input_df = pd.DataFrame(data, index=[0])

# Ensure input_df columns match X
input_df = input_df[X.columns]

st.write('**Input Data Values**')
st.write(input_df)

# Models setup
models = [
    ('LR', LogisticRegression()),
    ('IDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

# Model training and evaluation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=6)
predictions = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_result = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    predictions.append(f"{name}: {cv_result.mean():.6f} ({cv_result.std():.6f})")

parsed_data = [(item.split(":")[0], float(item.split(":")[1].split()[0])) for item in predictions]

# Sort data by accuracy
sorted_data = sorted(parsed_data, key=lambda x: x[1], reverse=True)
first_model, _ = sorted_data[0]

final_model = None
for model_name, model_instance in models:
    if model_name == first_model:
        final_model = model_instance
        break

# Train and predict 
if final_model:
    final_model.fit(X_train, y_train)
    prediction = final_model.predict(input_df)
    prediction_proba =final_model.predict_proba(input_df)
    df_prediction_proba =pd.DataFrame(prediction_proba)
    df_prediction_proba.columns =['Iris-setosa' ,'Iris-versicolor' , 'Iris-virginica']
    df_prediction_proba.rename(columns ={ 0: 'Iris-setosa' ,1: 'Iris-versicolor' , 2:'Iris-virginica'})
    
    st.write('**Prediction**')
    st.success(prediction)
else:
    st.error("No model selected.")
