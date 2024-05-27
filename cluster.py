#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


def eda(df):
    # Replace 'None' or NaN values with the median of each column
    df = df.fillna(df.median())
    
    # Calculating the percentage of null values for each column
    null_percentage = (df.isnull().mean() * 100).round(2)

    # Identify columns with more than 30% null values
    columns_to_drop = null_percentage[null_percentage > 30].index
    print('Columns having null values more than 30% are:', columns_to_drop, '\n\n')

    # Drop columns with more than 30% null values
    df1 = df.drop(columns=columns_to_drop)
    
    # Check if columns exist before replacing string values with float values
    columns_to_replace = ['Business Tax Rate', 'GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
    for col in columns_to_replace:
        if col in df1.columns:
            # Remove commas and replace '$' before converting to float
            df1[col] = df1[col].replace('[\$,]', '', regex=True).astype(float)
            # Replace 'None' or NaN values with the median of the column
            df1[col] = df1[col].fillna(df1[col].median())
    
    # Dropping No.of Records column in the dataset
    df1 = df1.drop(columns='Number of Records')
    
    # Converting the Country Column into numerical Values 
    label_encoder = LabelEncoder()
    df1['Country'] = label_encoder.fit_transform(df1['Country'])
    
    # Display a correlation heatmap
    st.write("### Correlation Heatmap:")
    corr_matrix = df1.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    st.pyplot()
    
    # Feature Scaling
    scaler = RobustScaler()
    # Scaling data
    scaled_data = scaler.fit_transform(df1)
    
    #Dimensionality Reduction
    data_tsne = TSNE(n_components=2).fit_transform(scaled_data)
    
    
    return data_tsne

def perform_clustering(tsne_data, num_clusters, clustering_algorithm):
    if clustering_algorithm == 'K-Means':
        # Instantiate the KMeans model
        model = KMeans(n_clusters=num_clusters, random_state=42)
    elif clustering_algorithm == 'Hierarchical':
        # Instantiate the AgglomerativeClustering model
        model = AgglomerativeClustering(n_clusters=num_clusters)

    # Fit the model to the t-SNE transformed data
    cluster_labels = model.fit_predict(tsne_data)

    # Add cluster labels to the original DataFrame
    tsne_data_with_clusters = pd.DataFrame(tsne_data, columns=['Dimension 1', 'Dimension 2'])
    tsne_data_with_clusters['Cluster'] = cluster_labels

    return tsne_data_with_clusters,cluster_labels

def evaluate_clusters(data, labels):
    # Silhouette Score
    silhouette_avg = silhouette_score(data, labels)
    st.write(f"Silhouette Score: {silhouette_avg}")

st.title('Upload Dataset')

uploaded_file = st.file_uploader('Choose a file to upload', type=['csv', 'xlsx'])

if uploaded_file is not None:
    if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader('Uploaded Data')
    st.write(df)

    # Apply the EDA function to modify the DataFrame
    data_tsne = eda(df)

    st.subheader('Data after EDA and Scaling')
    st.write(data_tsne)

    # Ask the user for the number of clusters
    num_clusters = st.slider("Select the number of clusters for t-SNE data:", min_value=2, max_value=10, value=3)

    # Ask the user to select the clustering algorithm
    clustering_algorithm = st.selectbox("Select the Clustering Algorithm:", ["K-Means", "Hierarchical"])

    # Perform clustering on t-SNE transformed data based on user selection
    tsne_data_with_clusters,cluster_labels = perform_clustering(data_tsne, num_clusters, clustering_algorithm)

    st.subheader(f'Data with Clusters ({clustering_algorithm})')
    st.write(tsne_data_with_clusters)

    # Display a scatter plot colored by clusters
    st.write(f"### Scatter Plot Colored by Clusters ({clustering_algorithm}):")
    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Cluster', data=tsne_data_with_clusters, palette='Set1')
    st.pyplot()
  
    # Evaluate the quality of clusters
    evaluate_clusters(data_tsne, cluster_labels)







