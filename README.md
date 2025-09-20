Title
Nonlinear Causal Discovery Based on Graph Autoencoder (GAE)

Description
This project provides a complete implementation of a nonlinear causal discovery method based on Graph Autoencoder (GAE), aiming to mine causal relationships between variables from observational data and support causal structure learning on various real-world datasets. The repository includes all datasets required for experiments and directly executable code, which can output causal adjacency matrices, causal strength matrices, and visualize causal discovery results.

Dataset Information
Data Location
All experimental datasets are stored in the data/dataset directory.
Included Datasets
768_1.csv: Standardized Pima Indians Diabetes Dataset, which is publicly available (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
LMCH_1.csv: Standardized Iraqi Diabetes Patient Dataset (from the Laboratory of Medical City Hospital),which is publicly available (https://data.mendeley.com/datasets/wj9rwkp9c2/1?utm_source=chatgpt.com).
2011-2020_1.csv: Standardized dataset from the National Health and Nutrition Examination Survey (NHANES), which is publicly available (https://www.cdc.gov/nchs/nhanes/Default.aspx).

Code Information
Code Structure
test/: Contains 3 experimental scripts, each corresponding to one dataset:
demo_real_data_768.py: Processes the Pima Indians Diabetes Dataset.
demo_real_data_LMCH.py: Processes the Iraqi Diabetes Patient Dataset.
demo_real_data_NA.py: Processes the NHANES Dataset.
recode/: Tool code:
gae.py: Core implementation of the Graph Autoencoder model.
causal_strength.py: Module for calculating causal strength.
utils.py: Utility functions (e.g., adjacency matrix visualization).
trainers/al_trainer.py: Model trainer for optimization.
models/model.py: Definition of the Graph Autoencoder architecture.
data/:
simulator.py: Tool for generating random causal Directed Acyclic Graphs (DAGs).
evaluation.py: Implementation of evaluation metrics for causal discovery results.
Core Functions
Load and preprocess data from CSV files.
Learn nonlinear causal relationships using the GAE model.
Output causal adjacency matrices and causal strength matrices.
Visualize causal discovery results.

Usage Instructions
Environment Preparation
Install required dependencies using pip: pip install torch pandas numpy scipy matplotlib networkx
Execution Steps
1.Clone or download the repository to your local machine.
2.Navigate to the experimental code directory: cd GAE/Section 3 code/test
3.Run the experimental script for the target dataset:
For the Pima Indians Diabetes Dataset: python demo_real_data_768.py
For the Iraqi Diabetes Patient Dataset: python demo_real_data_LMCH.py
For the NHANES Dataset: python demo_real_data_NA.py

Requirements 
Python 3.6 or higher
PyTorch (torch)
Pandas (pandas)
NumPy (numpy)
SciPy (scipy)
Matplotlib (matplotlib)
NetworkX (networkx)

Methods 
Data Preprocessing
1.Load data from CSV files and identify column names.
2.Remove label columns (non-feature variables, such as 'Outcome' or 'CLASS').
3.Convert the processed data into a float32-type NumPy array to adapt to the model.
Model Training
1.Initialize the GAE model with specified parameters (e.g., number of training epochs, device type).
2.Train the model through alternating optimization, including:
L1 regularization to enhance the sparsity of the causal graph.
Bidirectional edge penalty to ensure the learned graph is a Directed Acyclic Graph (DAG).
3.Monitor the loss function and constraints during training to ensure convergence.
Result Generation
1.Output the original causal adjacency matrix, indicating the existence of causal edges.
2.Calculate and output the causal strength matrix based on information entropy difference.
3.Visualize causal discovery results using adjacency matrix graphs.

Citation
If you use this code repository for research, please cite relevant literature on Graph Autoencoder-based causal discovery.

License and Contribution Guidelines
License
This project is open-source under the Apache License 2.0. For details, please refer to the LICENSE file.
Contribution Guidelines
Contributions to this project are welcome via submitting Issues or Pull Requests. All contributions will be reviewed before merging to ensure quality and consistency.
