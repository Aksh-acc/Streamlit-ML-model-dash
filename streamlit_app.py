import streamlit as st

st.title('Machine Learning prediction dash')
st.info('This is a machine learing model app where you can get the insights from the data and get the predictions along with')
df = pd.read_csv("iris.csv")
data = pd.DataFrame(df)
with st.expander("data"):
  st.write(data)
