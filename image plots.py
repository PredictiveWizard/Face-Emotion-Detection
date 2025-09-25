import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming you have loaded your data into a DataFrame
# Replace 'df' with the actual DataFrame
df = pd.read_csv('retail_sales_dataset.csv')

# Scatter Plot
plt.scatter(df['Age'], df['Total Amount'])
plt.xlabel('Age')
plt.ylabel('Total Amount')
plt.title('Scatter Plot')
plt.show()

# Histogram
plt.hist(df['Age'], bins=20, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram')

# Adding text annotations above the bars
hist, bin_edges = np.histogram(df['Age'], bins=20)
for i, value in enumerate(hist):
    plt.text(bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2, value,
             str(value), ha='center', va='bottom')

plt.show()

# Pie Chart
category_counts = df['Product Category'].value_counts()
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()