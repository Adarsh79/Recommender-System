import os
import streamlit as st

import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD


if os.path.exists("./ratings.csv"):
    df = pd.read_csv('ratings.csv', index_col=None)

with st.sidebar:
    st.title("Recommendation System")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling"])
    st.info("This application helps your find recommendation products for E-commerce platforms such as Amazon and Shopify")

if choice == "Upload":
    st.title("Upload your Dataset")
    file = st.file_uploader("Upload your Dataset")
    if file:
        df = pd.read_csv(file)
        df.to_csv("ratings.csv", index=None) # type: ignore
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis") 

if choice == "Modelling":
    chosen_target = st.selectbox("Choose the Target Column", df.columns)
    if st.button("Train Model"):
        df.drop(['timestamp'], axis=1, inplace=True)
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
        st.write(Recommend[0:24])