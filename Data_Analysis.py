import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import numpy as np




# URL of the CSV file
url = "https://storage.googleapis.com/courses_data/Assignment%20CSV/finance_liquor_sales.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(url)

# Display the first few rows of the DataFrame
print(df)

# Check the data types of each column in the DataFrame
data_types = df.dtypes

# Display the data types
print(data_types)

# Check for missing values in the DataFrame
missing_values = df.isnull().sum()

# Display the number of missing values for each column
print(missing_values)

# Calculate the percentage of missing values for each column
missing_percentage = (df.isnull().mean() * 100).round(2)

# Display the percentage of missing values for each column
print(missing_percentage)

# Set the threshold to the number of non-null values you want to keep
threshold = len(df.columns) - 1  # Set the threshold to len(df.columns) - 1 for more than 2 null values

# Drop rows with more than 2 null values
df_cleaned = df.dropna(axis=0,thresh=threshold)
print(df_cleaned)
# Drop rows with missing values
df_cleaned2 = df_cleaned.dropna(axis=1)

# Display the first few rows of the cleaned DataFrame
print(df_cleaned2)
df=df_cleaned2.copy()

# Check for missing values in the DataFrame
missing_values = df.isnull().sum()

# Display the number of missing values for each column
print(missing_values)

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter rows with date between 2016-01-01 and 2019-12-31
filtered_df = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2019-12-31')]

# Display the result
print(filtered_df.reset_index())

#df=filtered_df
# Group by 'zip_code' and 'item_number', and sum 'bottles_sold'
sum_by_zip_item = filtered_df.groupby(['zip_code', 'item_number'])['bottles_sold'].sum().reset_index()

# Find the index of the row with the maximum sum of 'bottles_sold' for each 'item_number' within each 'zip_code' group
idxmax_sum = sum_by_zip_item.groupby('zip_code')['bottles_sold'].idxmax()

# Select the corresponding rows from the original DataFrame
result_df = sum_by_zip_item.loc[idxmax_sum].reset_index()

# Display the result
print(result_df)

#from sklearn.preprocessing import MinMaxScaler

# Assuming result_df is your DataFrame
# Assuming 'bottles_sold', 'zip_code', and 'item_number' are columns in your DataFrame

# Create a color map with rainbow colors
colors = plt.cm.rainbow(np.linspace(0, 1, len(result_df)))

# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
result_df['zip_code_normalized'] = scaler.fit_transform(result_df[['zip_code']])

# Identify the top 5 rows based on 'bottles_sold'
top5 = result_df.nlargest(5, 'bottles_sold')

# Scatter plot with rainbow colors
plt.scatter(result_df['zip_code_normalized'], result_df['bottles_sold'], alpha=0.5, c=colors, s=50)

# Annotate the top 5 rows
for i, row in top5.iterrows():
    plt.annotate(f" {row['item_number']}",
                 (row['zip_code_normalized'], row['bottles_sold']),
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, color='red')

# Adding labels and title
plt.xlabel('Zip Code')
plt.ylabel('Bottles Sold')
plt.title('Scatter Plot of Bottles Sold by Zip Code (Group By Zip Code & item_number)')

# Display the plot
plt.show()

# Identify the top 5 rows based on 'bottles_sold'
top5 = result_df.nlargest(5, 'bottles_sold')

# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
result_df['zip_code_normalized'] = scaler.fit_transform(result_df[['zip_code']])

# Scatter plot with rainbow colors using Plotly Express
fig = px.scatter(result_df, x='zip_code_normalized', y='bottles_sold', opacity=0.95)

# Annotate the top 5 points with item numbers only
for i, row in top5.iterrows():
    fig.add_annotation(
        x=row['zip_code_normalized'],
        y=row['bottles_sold'],
        text=f"Item: {row['item_number']}",
        showarrow=True,
        arrowhead=3,
        arrowwidth=1,
        ax=0,
        ay=-30,
        font=dict(size=7, color='red')
    )

# Updating axis labels and title
fig.update_xaxes(title_text='Normalized Zip Code (0 to 70)')
fig.update_yaxes(title_text='Bottles Sold')
fig.update_layout(title_text='Scatter Plot of Bottles Sold by Normalized Zip Code (Group By Zip Code & item_number)',
                  showlegend=False)

# Show the plot
fig.show()

# Identify the top 5 rows based on 'bottles_sold'
top5 = result_df.nlargest(5, 'bottles_sold')

# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
result_df['zip_code_normalized'] = scaler.fit_transform(result_df[['zip_code']])

# Thermal scatter plot using Plotly Express
fig = px.scatter(result_df, 
                 x='zip_code_normalized', 
                 y='bottles_sold', 
                 color='bottles_sold', 
                 opacity=0.99,
                 color_continuous_scale='thermal',
                 color_discrete_map={'bottles_sold': 'black'},
                 size_max=20)

# Annotate the top 5 points with item numbers only
for i, row in top5.iterrows():
    fig.add_annotation(
        x=row['zip_code_normalized'],
        y=row['bottles_sold'],
        text=f"Item: {row['item_number']}",
        showarrow=True,
        arrowhead=3,
        arrowwidth=1,
        ax=0,
        ay=-30,
        font=dict(size=7, color='red')
    )

# Updating axis labels and title
fig.update_xaxes(title_text='Normalized Zip Code (0 to 70)')
fig.update_yaxes(title_text='Bottles Sold')
fig.update_layout(title_text='Thermal Scatter Plot of Bottles Sold by Normalized Zip Code (Group By Zip Code & item_number)',
                  showlegend=False)

# Show the plot
fig.show()

# Identify the top 5 rows based on 'bottles_sold'
top5 = result_df.nlargest(5, 'bottles_sold')

# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
result_df['zip_code_normalized'] = scaler.fit_transform(result_df[['zip_code']])

# Scatter plot with Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='zip_code_normalized', y='bottles_sold', data=result_df, alpha=0.5,color='blue')

# Annotate the top 5 points with item numbers only
for i, row in top5.iterrows():
    plt.annotate(f" {row['item_number']}",
                 (row['zip_code_normalized'], row['bottles_sold']),
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, color='red')

# Adding labels and title
plt.xlabel('Normalized Zip Code (0 to 70)')
plt.ylabel('Bottles Sold')
plt.title('Scatter Plot of Bottles Sold by Normalized Zip Code (Group By Zip Code & item_number)')

# Display the plot
plt.show()

# Assuming result_df is your DataFrame
# 'zip_code' is the column you want to group by

# Group by 'zip_code' and print only the 'zip_code' column
grouped_data1 = filtered_df.groupby('zip_code').size().reset_index(name='count')

# Print only the 'zip_code' column
print(grouped_data1)

# Group by 'zip_code' and 'item_number', and sum 'bottles_sold'
grouped_data = filtered_df.groupby(['zip_code','item_number'])['bottles_sold'].sum().reset_index()

# Print the grouped data
print(grouped_data)
# Specify the Excel file path
excel_file_path = 'grouped_data_for_theory1.xlsx'

# Export the grouped data to Excel
grouped_data.to_excel(excel_file_path, index=True)

print(f"Grouped data has been exported to {excel_file_path}")

from sklearn.preprocessing import MinMaxScaler

# Create a color map with rainbow colors
colors = plt.cm.rainbow(np.linspace(0, 1, len(grouped_data)))
# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
grouped_data['zip_code_normalized'] = scaler.fit_transform(grouped_data[['zip_code']])
# Identify the top 5 rows based on 'bottles_sold'
top5 = grouped_data.nlargest(5, 'bottles_sold')

# Scatter plot with rainbow colors
plt.scatter(grouped_data['zip_code_normalized'], grouped_data['bottles_sold'], alpha=0.5, c=colors,s=50)
for i, row in top5.iterrows():
    plt.annotate(f" {row['item_number']}",
                 (row['zip_code_normalized'], row['bottles_sold']),
                 textcoords="offset points", xytext=(0,5), ha='center', fontsize=7, color='red')

# Adding labels and title
plt.xlabel('Zip Code')
plt.ylabel('Bottles Sold')
plt.title('Scatter Plot of Bottles Sold by Zip Code (Group By Zip Code & item_number)')

# Display the plot
plt.show()

# Identify the top 5 rows based on 'bottles_sold'
top5 = grouped_data.nlargest(5, 'bottles_sold')

# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
grouped_data['zip_code_normalized'] = scaler.fit_transform(grouped_data[['zip_code']])

# Scatter plot with rainbow colors using Plotly Express
fig = px.scatter(grouped_data, x='zip_code_normalized', y='bottles_sold', opacity=0.95)

# Annotate the top 5 points with item numbers only
for i, row in top5.iterrows():
    fig.add_annotation(
        x=row['zip_code_normalized'],
        y=row['bottles_sold'],
        text=f"Item: {row['item_number']}",
        showarrow=True,
        arrowhead=3,
        arrowwidth=1,
        ax=0,
        ay=-30,
        font=dict(size=7, color='red')
    )

# Updating axis labels and title
fig.update_xaxes(title_text='Normalized Zip Code (0 to 70)')
fig.update_yaxes(title_text='Bottles Sold')
fig.update_layout(title_text='Scatter Plot of Bottles Sold by Normalized Zip Code (Group By Zip Code & item_number)',
                  showlegend=False)

# Show the plot
fig.show()

# Identify the top 5 rows based on 'bottles_sold'
top5 = grouped_data.nlargest(5, 'bottles_sold')

# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
grouped_data['zip_code_normalized'] = scaler.fit_transform(grouped_data[['zip_code']])

# Thermal scatter plot using Plotly Express
fig = px.scatter(grouped_data, 
                 x='zip_code_normalized', 
                 y='bottles_sold', 
                 color='bottles_sold', 
                 opacity=0.99,
                 color_continuous_scale='thermal',
                 color_discrete_map={'bottles_sold': 'black'},
                 size_max=20)

# Annotate the top 5 points with item numbers only
for i, row in top5.iterrows():
    fig.add_annotation(
        x=row['zip_code_normalized'],
        y=row['bottles_sold'],
        text=f"Item: {row['item_number']}",
        showarrow=True,
        arrowhead=3,
        arrowwidth=1,
        ax=0,
        ay=-30,
        font=dict(size=7, color='red')
    )

# Updating axis labels and title
fig.update_xaxes(title_text='Normalized Zip Code (0 to 70)')
fig.update_yaxes(title_text='Bottles Sold')
fig.update_layout(title_text='Thermal Scatter Plot of Bottles Sold by Normalized Zip Code (Group By Zip Code & item_number)',
                  showlegend=False)

# Show the plot
fig.show()

# Identify the top 5 rows based on 'bottles_sold'
top5 = grouped_data.nlargest(5, 'bottles_sold')

# Normalize the 'zip_code' values to be between 0 and 75
scaler = MinMaxScaler(feature_range=(0, 70))
grouped_data['zip_code_normalized'] = scaler.fit_transform(grouped_data[['zip_code']])

# Scatter plot with Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='zip_code_normalized', y='bottles_sold', data=grouped_data, alpha=0.5,color='blue')

# Annotate the top 5 points with item numbers only
for i, row in top5.iterrows():
    plt.annotate(f" {row['item_number']}",
                 (row['zip_code_normalized'], row['bottles_sold']),
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, color='red')

# Adding labels and title
plt.xlabel('Normalized Zip Code (0 to 70)')
plt.ylabel('Bottles Sold')
plt.title('Scatter Plot of Bottles Sold by Normalized Zip Code (Group By Zip Code & item_number)')

# Display the plot
plt.show()

# Display the filter data
print(filtered_df.reset_index())

# Group by 'store_name' and sum the 'sale_dollars' column
grouped_df = filtered_df.groupby('store_name')['sale_dollars'].sum().reset_index()

# Display the result
print(grouped_df)

# Sort in descending order based on the sum of 'sale_dollars'
grouped_df_sorted = grouped_df.sort_values(by='sale_dollars', ascending=False)

# Display the result
print(grouped_df_sorted)

# Keep only the first 15 rows
top_15_rows = grouped_df_sorted.head(15)

# Display the result
print(top_15_rows)

# Find the sum of the 'sale_dollars' column as a float
total_sale_dollars = float(grouped_df_sorted['sale_dollars'].sum())

# Display the result
print(total_sale_dollars)

# Use .loc to create a new column by dividing 'sale_dollars' by the total sale dollars
top_15_rows.loc[:, 'sale_dollars_normalized'] = ((top_15_rows['sale_dollars'] / total_sale_dollars) * 100).copy()

# Display the result
print(top_15_rows)

# Sort the DataFrame by 'sale_dollars_normalized' in descending order
top_15_rows_sorted = top_15_rows.sort_values(by='sale_dollars_normalized', ascending=True)

# Create a horizontal bar plot
fig, ax = plt.subplots()
bars = ax.barh(top_15_rows_sorted['store_name'], top_15_rows_sorted['sale_dollars_normalized'])
plt.xlabel('Sale Dollars Normalized (%)')
plt.ylabel('Store Name')
plt.title('Horizontal Bar Plot of Sale Dollars Normalized')

# Add percentage numbers to the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.3f}%',
             va='center', ha='left')

# Display the plot
plt.show()

# Sort the DataFrame by 'sale_dollars_normalized' in descending order
top_15_rows_sorted = top_15_rows.sort_values(by='sale_dollars_normalized', ascending=False)

# Create a horizontal bar plot with Seaborn
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
rainbow_colors = sns.color_palette('rainbow', n_colors=len(top_15_rows_sorted))
ax = sns.barplot(x='sale_dollars_normalized', y='store_name', data=top_15_rows_sorted, palette=rainbow_colors)

# Add percentage numbers to the bars
for p in ax.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height() / 2, f'{width:.3f}%', va='center', ha='left')

# Set labels and title
plt.xlabel('Sale Dollars Normalized (%)')
plt.ylabel('Store Name')
plt.title('Horizontal Bar Plot of Sale Dollars Normalized')

# Display the plot
plt.show()

# Sort the DataFrame by 'sale_dollars_normalized' in descending order
top_15_rows_sorted = top_15_rows.sort_values(by='sale_dollars_normalized', ascending=True)

# Create a horizontal bar plot with Plotly
fig = px.bar(top_15_rows_sorted, 
             x='sale_dollars_normalized', 
             y='store_name',
             text='sale_dollars_normalized',
             labels={'sale_dollars_normalized': 'Sale Dollars Normalized (%)'},
             title='Horizontal Bar Plot of Sale Dollars Normalized with Plotly',
             color='sale_dollars_normalized',
             color_continuous_scale='Rainbow')

# Adjust the x-axis width
fig.update_layout(xaxis=dict(tickvals=[0, 25, 50, 75, 100], range=[0, 22]))
# Display percentage numbers on the bars
fig.update_traces(texttemplate='%{text:.3f}%', textposition='outside')

# Show the figure
fig.show()