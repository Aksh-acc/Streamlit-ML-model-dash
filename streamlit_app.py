import streamlit as st
import pandas as pd
st.title('Machine Learning prediction dash')
st.info('This is a machine learing model app where you can get the insights from the data and get the predictions along with')
df = pd.read_csv('https://raw.githubusercontent.com/Aksh-acc/Streamlit-ML-model-dash/refs/heads/master/Iris.csv')
with st.expander("data"):
  st.write(df)
st.write('**X vlaue**')
X = df.drop('Species' ,axis =1)
X
st.write('**Y vlaue**')
y = df.Species
y
