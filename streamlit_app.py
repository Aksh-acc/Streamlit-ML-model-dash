import streamlit as st
import pandas as pd
st.title('Machine Learning prediction dash')
st.info('This is a machine learing model app where you can get the insights from the data and get the predictions along with')
df = pd.read_csv(Iris.csv)
with st.expander("data"):
  st.write(df)
