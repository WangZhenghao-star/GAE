import torch
from recode.utils import output_adj
from recode.gae import GAE
import pandas as pd
import numpy as np
from recode.causal_strength import adj_cs

# Load the dataset and ensure the first row is recognized as column names
X = pd.read_csv("../data/dataset/LMCH_1.csv")

# Check the column names and confirm there are no unexpected strings
print(X.columns)

# If the 'CLASS' column exists and you want to delete it
if 'CLASS' in X.columns:
    X = X.drop(columns=['CLASS'])
else:
    print("列 'CLASS' 不存在。")

# Convert the DataFrame to a NumPy array and ensure the data type is float32
X_np = X.values.astype(np.float32)  # Convert the data to a NumPy array

# Initialize the GAE model
gae = GAE(epochs=10, device_type="cpu")

# Train the model
gae.learn(X_np)  # Pass the NumPy array directly

# Output the causal adjacency matrix
print(gae.causal_matrix)

# Output the causal strength adjacency matrix (if needed)
print(adj_cs(gae.causal_matrix, X_np))

# Use the output_adj function from recode.utils to output the adjacency matrix
output_adj(gae.causal_matrix)
