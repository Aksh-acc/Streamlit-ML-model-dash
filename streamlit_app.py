import streamlit as st

st.title('Machine Learning prediction dash')
st.info('This is a machine learing model app where you can get the insights from the data and get the predictions along with') 
with st.expander("data"):
  st.write("iris.csv")
