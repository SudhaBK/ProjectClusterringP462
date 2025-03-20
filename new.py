import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Custom CSS for background and styling
def add_background():
    st.markdown(
        f"""
        <style>
        /* Background Image */
        .stApp {{
            background: url("https://thumbs.dreamstime.com/b/intersection-money-global-economy-shaping-financial-landscapes-worldwide-intersection-money-global-economy-292671686.jpg") no-repeat center center fixed; 
            background-size: cover;
        }}
        /* Text Styling */
        h1, h2, h3, h4, h5, h6 {{
            color: white;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }}
        p, label, .stMarkdown {{
            color: white;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
        }}
        .stSidebar {{
            background-color: rgba(0, 0, 0, 0.7);
        }}
        .css-1p1n3ar {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the background CSS
add_background()

# Load the pre-trained KMeans model
with open('./kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load and clean the dataset
df = pd.read_csv('./World_development_mesurement.csv')
df = df.applymap(lambda x: str(x).replace('%', ''))
df = df.apply(pd.to_numeric, errors='coerce')

# Streamlit app title and description
st.title('🌍 Global Development Clustering App 🌍')
st.markdown("""
This app uses **KMeans clustering** to analyze and group countries based on development metrics, 
such as GDP, CO2 emissions, and life expectancy.  
Use this tool to explore patterns in global development!
""")

# Sidebar for user interaction
st.sidebar.header('User Input')
cluster = st.sidebar.selectbox('Select a Cluster:', range(5), format_func=lambda x: f'Cluster {x + 1}')

# Display the Cluster Color Legend in the sidebar
st.sidebar.subheader('Cluster Color Legend')
legend_info = {
    0: 'Strong economic development and high life expectancy (Blue)',
    1: 'Moderate economic development and medium life expectancy (Orange)',
    2: 'Weak economic development and low life expectancy (Green)',
    3: 'Unique development profiles (Red)',
    4: 'Other development profiles (Purple)'
}
for i, desc in legend_info.items():
    st.sidebar.write(f'* **Cluster {i + 1}**: {desc}')

# Display cluster details
st.subheader(f'Details for Cluster {cluster + 1}')
cluster_data = df[kmeans.labels_ == cluster]
st.write("### Key Characteristics")
st.write(cluster_data.describe())

# Visualizing all clusters with common scatter plot
st.write("### Visualizing All Clusters")
fig, ax = plt.subplots()
colors = ['blue', 'orange', 'green', 'red', 'purple']
scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=[colors[i] for i in kmeans.labels_], alpha=0.7, edgecolor='k')
ax.set_xlabel('GDP')
ax.set_ylabel('CO2 Emissions')
st.pyplot(fig)

# Visualizing the focused view of the selected cluster (only black dots for the selected cluster)
st.write(f"### Focused View of Cluster {cluster + 1}")
fig, ax = plt.subplots()

# Show only the selected cluster in black
selected_cluster_data = df[kmeans.labels_ == cluster]
ax.scatter(selected_cluster_data.iloc[:, 0], selected_cluster_data.iloc[:, 1], c='black', label=f'Selected Cluster {cluster + 1}', edgecolor='white')

# Add label to show that black dots are for the selected cluster
ax.text(
    0.5, 0.95, 'Black Dots Represent Selected Cluster', 
    ha='center', va='center', transform=ax.transAxes, color='white', fontsize=12, fontweight='bold'
)

ax.set_xlabel('GDP')
ax.set_ylabel('CO2 Emissions')
ax.legend()
st.pyplot(fig)
