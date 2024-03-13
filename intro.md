# Product Recommendation App

Welcome to our Product Recommendation App! This application is designed to help you find relevant product recommendations for e-commerce platforms like Amazon and Shopify. It utilizes a powerful machine learning model to analyze customer ratings and generate personalized recommendations based on user preferences and product similarities.

## Navigation

- **Upload**: Upload your dataset containing product ratings and information. The app supports CSV files and will guide you through the upload process.
- **Profiling**: Explore and analyze your dataset using various data profiling techniques. This feature helps you understand the structure, quality, and statistical properties of your data, allowing you to identify potential issues or areas for improvement.
- **Modelling**: Train a machine learning model to generate product recommendations based on your data. The app uses a collaborative filtering approach, specifically the Truncated Singular Value Decomposition (SVD) algorithm, to find patterns in user-product interactions and predict relevant recommendations.

## Machine Learning Model

The Product Recommendation App employs the Truncated SVD algorithm, a matrix factorization technique, to uncover latent features and similarities between users and products. This algorithm decomposes the user-product rating matrix into lower-dimensional matrices, capturing the underlying patterns and preferences.

By analyzing the relationships between users and items, the model can predict how a user might rate an unrated product. This information is then used to suggest products that are likely to be of interest to the user based on their past ratings and the ratings of other users with similar preferences.

## Data Cleaning and Preprocessing

Before training the machine learning model, the app performs several data cleaning and preprocessing steps to ensure optimal performance and accurate recommendations:

1. **Timestamp Removal**: The 'timestamp' column, if present, is dropped from the dataset as it is not relevant for the recommendation process.
2. **Filtering**: The app filters the dataset to include only products that have received a minimum number of ratings (e.g., 50 ratings). This step helps to eliminate products with insufficient data and improve the quality of recommendations.
3. **Sampling**: To optimize computational performance, the app takes a subset of the filtered data (e.g., the first 10,000 rows) for modeling.
4. **Pivot Table**: The ratings data is transformed into a pivot table format, with users as rows, products as columns, and ratings as values. This structure is suitable for the Truncated SVD algorithm.

After these steps, the cleaned and preprocessed data is ready for the machine learning model to generate accurate and reliable product recommendations.

## Getting Started

To begin using the Product Recommendation App, simply upload your dataset containing product ratings and information. The app will guide you through the necessary steps to profile your data, train the machine learning model, and generate personalized product recommendations.

Let's get started and discover the power of data-driven product recommendations!