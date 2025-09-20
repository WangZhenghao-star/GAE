# Import PyTorch library for tensor operations and deep learning capabilities
import torch
# Import output_adj function from recode.utils for adjacency matrix output handling
from recode.utils import output_adj
# Import GAE (Graph Autoencoder) class from recode.gae for causal discovery
from recode.gae import GAE
# Import pandas library for data manipulation and analysis
import pandas as pd
# Import numpy library for numerical computing
import numpy as np
# Import adj_cs function from recode.causal_strength for calculating causal strength
from recode.causal_strength import adj_cs

# Load the dataset from CSV file, with first row treated as column headers
X = pd.read_csv("../data/dataset/2011-2020_1.csv")

# Print column names to inspect data structure and check for unexpected columns
print(X.columns)

# Check if 'CLASS' column exists; remove it if present
if 'CLASS' in X.columns:
    X = X.drop(columns=['CLASS'])
else:
    # Print message if 'CLASS' column is not found in the dataset
    print("The column 'CLASS' does not exist.")

# Convert pandas DataFrame to NumPy array and cast to float32 for model compatibility
X_np = X.values.astype(np.float32)  # Convert data to NumPy array

# Initialize GAE model with 10 training epochs, configured to run on CPU
gae = GAE(epochs=10, device_type="cpu")

# Train the GAE model using the preprocessed NumPy array
gae.learn(X_np)  # Pass the NumPy array directly to the learning method

# Print the causal adjacency matrix learned by the trained GAE model
print(gae.causal_matrix)

# Calculate and print the causal strength adjacency matrix (when required)
print(adj_cs(gae.causal_matrix, X_np))

# Utilize output_adj function from recode.utils to output the adjacency matrix
output_adj(gae.causal_matrix)