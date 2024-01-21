# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 18:06:15 2024

@author: sam jacob
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit 
# Load data
file_path = "C://Users//sam jacob//OneDrive//Desktop//cluster//Clustering Data (1).xlsx"
df = pd.read_excel(file_path)

# Explore the data
print(df.head())

# Assuming your DataFrame is named 'df'
# Keep the non-year columns as identifiers, and melt the rest
melted_df = pd.melt(df, id_vars=['Country Code', 'Series Code', 'Country Name', 'Series Name'],
                    var_name='Year', value_name='Value')

# Convert the 'Year' column to numeric
melted_df['Year'] = melted_df['Year'].str.extract('(\d+)', expand=False).astype(float)

# Display the melted DataFrame
print(melted_df.head())

# Assuming df is your DataFrame
# Convert the year columns to a single 'Year' column
melted_df = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
                    var_name='Year', value_name='Value')

# Filter the melted DataFrame for the specified series names
selected_series_names = [
    'GDP (current US$)',
    'CO2 emissions (metric tons per capita)',
    'CO2 emissions (kg per 2015 US$ of GDP)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Forest area (% of land area)',
    'Population growth (annual %)'
]

melted_df = melted_df[melted_df['Series Name'].isin(selected_series_names)]

# Pivot the DataFrame to create separate columns for each series
reshaped_df = melted_df.pivot_table(index=['Country Name', 'Country Code', 'Year'],
                                    columns='Series Name', values='Value', aggfunc='first').reset_index()

# Display the reshaped DataFrame
reshaped_df.head(200) 

# Convert the 'Year' column to string and then extract numeric values
reshaped_df['Year'] = pd.to_numeric(reshaped_df['Year'].astype(str).str.extract('(\d+)', expand=False))
# Selecting numerical columns for clustering
numerical_columns = ['CO2 emissions (kg per 2015 US$ of GDP)',
                     'CO2 emissions (metric tons per capita)',
                     'Forest area (% of land area)',
                     'GDP (current US$)',
                     'Population growth (annual %)',
                     'Renewable energy consumption (% of total final energy consumption)']
# Extracting only numerical columns for clustering
clustering_data = reshaped_df[numerical_columns]

# Convert the 'Value' column to numeric (replace non-numeric values with NaN)
clustering_data = clustering_data.apply(pd.to_numeric, errors='coerce')

# Handling missing values (replace NaN with 0 for simplicity, you may choose a different strategy)
clustering_data.fillna(0, inplace=True)

# Standardize the data for better clustering results
clustering_data_standardized = (clustering_data - clustering_data.mean()) / clustering_data.std()

# Apply KMeans clustering (explicitly setting n_init)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
reshaped_df['Cluster'] = kmeans.fit_predict(clustering_data_standardized)
# Convert the 'Cluster' column to string
reshaped_df['Cluster'] = reshaped_df['Cluster'].astype(str)
plot_data = reshaped_df[['GDP (current US$)', 'CO2 emissions (metric tons per capita)', 'Cluster']].dropna()
print(reshaped_df.dtypes) 

# Convert specific columns in reshaped_df to numeric, coercing errors to NaN
cols_to_convert = ['CO2 emissions (kg per 2015 US$ of GDP)',
                   'CO2 emissions (metric tons per capita)',
                   'Forest area (% of land area)',
                   'GDP (current US$)',
                   'Population growth (annual %)',
                   'Renewable energy consumption (% of total final energy consumption)']

reshaped_df[cols_to_convert] = reshaped_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Convert 'Cluster' column in plot_data to numeric
plot_data['Cluster'] = pd.to_numeric(plot_data['Cluster'], errors='coerce')

# Ensure that the plotting columns are numeric
plot_columns = ['GDP (current US$)', 'CO2 emissions (metric tons per capita)']
plot_data[plot_columns] = plot_data[plot_columns].apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values in the plotting columns or the 'Cluster' column
plot_data.dropna(subset=plot_columns + ['Cluster'], inplace=True)

# Visualize clustering results with a scatter plot
plt.scatter(plot_data['GDP (current US$)'], plot_data['CO2 emissions (metric tons per capita)'],
            c=plot_data['Cluster'], cmap='viridis')
plt.title('Clustering Results')
plt.xlabel('GDP (current US$)')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.show()  
# Fit KMeans clustering model
kmeans = KMeans(n_clusters=3, random_state=42)
reshaped_df['Cluster'] = kmeans.fit_predict(clustering_data_standardized)

# Visualize clustering results (2D plot)
plt.scatter(reshaped_df['GDP (current US$)'], reshaped_df['CO2 emissions (metric tons per capita)'],
            c=reshaped_df['Cluster'], cmap='viridis', alpha=0.7)  # Use alpha to make points semi-transparent

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

plt.title('Clustering Results with Cluster Centers')
plt.xlabel('GDP (current US$)')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.legend()
plt.show()  

# Identify columns with non-numeric values
non_numeric_columns = reshaped_df.select_dtypes(exclude=['number']).columns

# Convert non-numeric columns to numeric, coercing errors
reshaped_df[non_numeric_columns] = reshaped_df[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

# Compute mean values for each cluster
cluster_means = reshaped_df.groupby('Cluster').mean()

# Display the mean values for each cluster
print(cluster_means) 

# Define the model function
def exponential_growth(x, a, b, c):
    return a * np.exp(b * (x - 1970)) + c

# Extract relevant data and clean the data
cleaned_df = reshaped_df.dropna(subset=['Year', 'GDP (current US$)'])
x_data = cleaned_df['Year']
y_data = cleaned_df['GDP (current US$)']

# Initial guess for parameters
initial_guess = [1e12, 0.02, 1e11]

# Fit the model to the data
params, covariance = curve_fit(exponential_growth, x_data, y_data, p0=initial_guess)

# Predictions for future years
future_years = np.arange(2022, 2042, 1)
predicted_values = exponential_growth(future_years, *params)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Original Data', color='blue')
plt.plot(future_years, predicted_values, label='Exponential Growth Fit', color='red')
plt.title('Exponential Growth Fit for GDP over Years')
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')
plt.legend()
plt.show()
# Display the parameters obtained from the fitted model
print("Fitted Parameters:")
print(f"a (amplitude): {params[0]}")
print(f"b (growth rate): {params[1]}")
print(f"c (offset): {params[2]}") 
# Define the exponential growth model
def exponential_growth(x, a, b, c):
    return a * np.exp(b * (x - 2021)) + c

# Generate synthetic data for demonstration
np.random.seed(42)
x_data = np.arange(2010, 2022)
y_data = 1e12 * np.exp(0.02 * (x_data - 2021)) + 1e11 + np.random.normal(scale=1e10, size=len(x_data))

# Fit the model to the data with increased maxfev
initial_guess = [1e12, 0.02, 1e11]
params, covariance = curve_fit(exponential_growth, x_data, y_data, p0=initial_guess, maxfev=5000)

# Predictions for future years
future_years = np.arange(2022, 2042, 1)
predicted_values = exponential_growth(future_years, *params)

# Plotting the original data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(future_years, predicted_values, label='Fitted Curve', color='red')

plt.title('Exponential Growth Model Fitting')
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')
plt.legend()
plt.show()  

# Group by Cluster and compute mean values for reshaped_df
cluster_means = reshaped_df.groupby('Cluster').mean()

# Display mean values for each cluster in reshaped_df
print(cluster_means)  


# Set the DPI for the figure
dpi = 100

# Set a larger plot size (50x50 inches for 5000x5000 pixels at 100 DPI)
plt.figure(figsize=(50, 50), dpi=dpi)

# Adjust the layout
plt.subplots_adjust(wspace=0.2, hspace=0.8)

# Select a subset of columns for the pair plot to reduce clutter
columns_to_plot = ['GDP (current US$)', 'CO2 emissions (metric tons per capita)', 
                   'Population growth (annual %)', 'Renewable energy consumption (% of total final energy consumption)']
subset_df = reshaped_df[columns_to_plot + ['Cluster']]

# Create the pair plot
pair_plot = sns.pairplot(
    subset_df, 
    hue='Cluster', 
    diag_kind='kde', 
    plot_kws={'alpha': 0.5, 's': 50}
)

# Adjust the title of the pair plot
plt.suptitle('Pairplot of Selected Variables by Cluster', y=1.02)

# Save the pair plot to a file (e.g., PNG)
pair_plot.savefig('pair_plot.png', dpi=dpi)

plt.show()


# Load the Excel file into a pandas DataFrame
file_path = "C://Users//sam jacob//OneDrive//Desktop//cluster//reshaped_data.xlsx"
df = pd.read_excel(file_path)

# Display the first few rows of the DataFrame to understand the structure
print(df.head())

# Display the first few rows of the DataFrame to understand the structure
print(df.head())

# Display basic information about the DataFrame
print("Column Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Check for any potential errors or anomalies in the data
print("\nData Summary:")
print(df.describe())

# Check for unique values in categorical columns
for column in df.select_dtypes(include=['object']).columns:
    print(f"\nUnique values in {column}:")
    print(df[column].unique())


# Check for any potential errors or anomalies in the data
print("\nData Summary:")
print(df.describe())
# Check for unique values in categorical columns
for column in df.select_dtypes(include=['object']).columns:
    print(f"\nUnique values in {column}:")
    print(df[column].unique())

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
sns.boxplot(data=df[numeric_columns])
plt.title('Boxplot of Numerical Columns')
plt.show()

# Convert columns to appropriate data types
df['CO2 emissions (metric tons per capita)'] = pd.to_numeric(df['CO2 emissions (metric tons per capita)'], errors='coerce')
df['Forest area (% of land area)'] = pd.to_numeric(df['Forest area (% of land area)'], errors='coerce')
df['GDP (current US$)'] = pd.to_numeric(df['GDP (current US$)'], errors='coerce')
df['Population growth (annual %)'] = pd.to_numeric(df['Population growth (annual %)'], errors='coerce')
df['Renewable energy consumption (% of total final energy consumption)'] = pd.to_numeric(df['Renewable energy consumption (% of total final energy consumption)'], errors='coerce')

print("\nUpdated Data Types:")
print(df.dtypes)
# Handling missing values for numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())



print("\nUpdated Missing Values:")
print(df.isnull().sum())


# Select relevant features for clustering
features = df[['CO2 emissions (kg per 2015 US$ of GDP)',
               'CO2 emissions (metric tons per capita)',
               'Forest area (% of land area)',
               'GDP (current US$)',
               'Population growth (annual %)',
               'Renewable energy consumption (% of total final energy consumption)']]

# Convert columns to appropriate data types
features = features.apply(pd.to_numeric, errors='coerce')


# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Choose the number of clusters (you may need to tune this based on your data)
n_clusters = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)


# Visualize the clusters using PCA (Principal Component Analysis)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)
df['PC1'] = principal_components[:, 0]
df['PC2'] = principal_components[:, 1]

# Plot the clusters in a 2D space
plt.figure(figsize=(10, 6))
plt.scatter(df['PC1'], df['PC2'], c=df['cluster'], cmap='viridis', alpha=0.5)
plt.title('K-Means Clustering of Countries')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Display the countries in each cluster
for cluster_id in range(n_clusters):
    cluster_countries = df[df['cluster'] == cluster_id]['Country Name'].unique()
    print(f"\nCountries in Cluster {cluster_id + 1}:")
    print(cluster_countries)

#Comparison analysis
# Select countries for comparison
countries_to_compare = ['United States', 'Brazil', 'China']

# Display information for the selected countries
selected_countries_data = df[df['Country Name'].isin(countries_to_compare)][['Country Name', 'cluster'] + features.columns.tolist()]
print(selected_countries_data)
# Calculate silhouette score for different cluster numbers
silhouette_scores = []

for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, df['cluster'])
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores for different cluster numbers
plt.figure(figsize=(10, 6))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Cluster Numbers') 
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose the number of clusters based on the silhouette score analysis
best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Perform K-Means clustering with the best number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Display information for the selected countries
selected_countries_data = df[df['Country Name'].isin(countries_to_compare)][['Country Name', 'cluster'] + features.columns.tolist()]
print(selected_countries_data)


# Display statistics for each country within their clusters
selected_countries_stats = selected_countries_data.groupby(['Country Name', 'cluster']).describe().stack(0)

# Display the result
pd.set_option('display.max_columns', None)
print(selected_countries_stats)
