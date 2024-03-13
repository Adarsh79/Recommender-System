import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


# Set the page configuration
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Load intro.md
with open("intro.md", "r") as f:
    intro_md = f.read()
st.markdown(intro_md)

# Load data if available
if os.path.exists("./ratings.csv"):
    df = pd.read_csv('ratings.csv', index_col=None)

with st.sidebar:
    st.title("Navigation")
    choice = st.radio("", ["Upload", "Profiling", "Modelling"])

if choice == "Upload":
    st.title("Upload your Dataset")
    file = st.file_uploader("Upload your Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df.to_csv("ratings.csv", index=False)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if st.button("Generate Profiling Report"):
        if "df" in locals():
            profile = df.describe()
            st.write(profile, use_container_width=True)
        else:
            st.warning("Please upload a dataset first.")

if choice == "Modelling":
    st.title("Product Recommendation Model")
    if "df" not in locals():
        st.warning("Please upload a dataset first.")
    else:
        chosen_target = st.selectbox("Choose the Target Column", df.columns)
        if st.button("Train Model"):
            df = df.drop(['timestamp'], axis=1, errors='ignore')
            subset_1 = df.groupby("productId").filter(lambda x: x['Rating'].count() >= 50)
            subset_2 = subset_1.head(10000)
            ratings_matrix = subset_1.pivot_table(values="Rating", index='userId', columns='productId', fill_value=0)
            X = ratings_matrix.T
            SVD = TruncatedSVD(n_components=10)
            decomposed_matrix = SVD.fit_transform(X)
            correlation_matrix = np.corrcoef(decomposed_matrix)
            product = "B00000K135"
            product_names = list(X.index)
            product_ID = product_names.index(product)
            correlation_product_ID = correlation_matrix[product_ID]
            Recommend = list(X.index[correlation_product_ID > 0.65])
            Recommend.remove(product)
            st.write(f"Recommended products for '{product}': {', '.join(Recommend[:24])}")