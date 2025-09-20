# Import the PyTorch library for tensor operations and deep learning functionalities
import torch
# Import the output_adj function from recode.utils for handling adjacency matrix output
from recode.utils import output_adj
# Import the GAE (Graph Autoencoder) class from recode.gae for causal discovery
from recode.gae import GAE
# Import pandas library for data manipulation and analysis
import pandas as pd
# Import numpy library for numerical computations
import numpy as np
# Import the adj_cs function from recode.causal_strength for calculating causal strength
from recode.causal_strength import adj_cs

# Load the dataset from a CSV file, ensuring the first row is treated as column headers
X = pd.read_csv("../data/dataset/768_1.csv")

# Print the column names of the DataFrame to verify data structure and check for unexpected columns
print(X.columns)

# Check if the 'Outcome' column exists in the DataFrame; if so, remove it
if 'Outcome' in X.columns:
    X = X.drop(columns=['Outcome'])
else:
    # Print a message if the 'Outcome' column is not found in the DataFrame
    print("The column 'Outcome' does not exist.")

# Convert the pandas DataFrame to a NumPy array and cast the data type to float32 for model compatibility
X_np = X.values.astype(np.float32)  # Convert the data to a NumPy array

# Initialize the GAE model with 10 training epochs and set to run on CPU
gae = GAE(epochs=10, device_type="cpu")

# Train the GAE model using the preprocessed NumPy array
gae.learn(X_np)  # Pass the NumPy array directly

# Print the learned causal adjacency matrix from the trained GAE model
print(gae.causal_matrix)

# Calculate and print the causal strength adjacency matrix using the causal matrix and original data
print(adj_cs(gae.causal_matrix, X_np))

# Use the output_adj function to process and output the causal adjacency matrix
output_adj(gae.causal_matrix)