import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# We are Loading the pre-trained KMeans_model from the pickle file here
with open('./kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# As usual Load the dataset
df = pd.read_csv('./World_development_mesurement.csv')

# As there were some symbols so Cleaning the data here
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# NOw we need to Create our Streamlit app naming as "Clusterring App"
st.title('Clustering App')
st.write('This app uses KMeans clustering to group countries based on their development metrics.')

# Select any cluster to view its characteristics here
st.write('Select a cluster to view its characteristics:')
cluster = st.selectbox('Cluster', range(5))

# It will Display here the characteristics of the selected cluster(any choosen)
st.write('Cluster', cluster)
st.write('Characteristics:')
st.write(df[kmeans.labels_ == cluster].describe())

# Here we r Creating a placeholder for the plot that will update dynamically as the cluster selection changes
plot_placeholder = st.empty()

# Display a scatter plot as per our choice or selected cluster
with plot_placeholder.container():
    fig, ax = plt.subplots()
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=[colors[i] for i in kmeans.labels_])
    ax.set_xlabel('Feature 1- GDP')
    ax.set_ylabel('Feature 2 - CO2 Emissions')
    plt.colorbar(scatter)
    st.pyplot(fig)

    # Here the plot will change as we choose clusters , so update the plot  when the selected cluster changes
    if st.button('Update Plot'):
        fig, ax = plt.subplots()
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        scatter = ax.scatter(df.iloc[kmeans.labels_ == cluster, 0], df.iloc[kmeans.labels_ == cluster, 1], c='black')
        ax.set_xlabel('Feature 1- GDP')
        ax.set_ylabel('Feature 2- CO2 Emission')
        st.pyplot(fig)

# Here just Printing to show the meaning of each color used to group countries into clusters.
st.write('Color Legend:')
st.write('* Cluster 0 (Blue) : Strong economic development and high life expectancy')
st.write('* Cluster 1 (Orange): Moderate economic development and medium life expectancy')
st.write('* Cluster 2 (Green): Weak economic development and low life expectancy')
st.write('* Cluster 3 (Red): Unique development profiles')
st.write('* Cluster 4 (Purple): Other development profiles')