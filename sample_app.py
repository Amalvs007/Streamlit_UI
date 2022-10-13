import streamlit as st 
import pandas as pd 
import numpy as np

# Display Text

st.title("Luminar Technolab")
st.header("Luminar Technolab")
st.text("Luminar Technolab")

# Display Table

df = pd.DataFrame(
   np.random.randn(10, 5),
   columns=('col %d' % i for i in range(5)))

st.table(df)

# Display Dataframe

#st.write(df)
st.dataframe(df)