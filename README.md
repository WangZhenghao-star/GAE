Nonlinear Causal Discovery Method Implementation Based on Graph Autoencoder (GAE)
Description
This project provides a complete implementation of a nonlinear causal discovery method based on Graph Autoencoder (GAE), designed to mine causal relationships between variables from observational data. The code repository includes all datasets required for experiments and directly executable experimental code, supporting causal structure learning on different types of real-world datasets and enabling the output of causal adjacency matrices and causal strength matrices.
Dataset Information
Data Location: All experimental datasets are stored in the data/dataset folder.
Included Datasets:
768_1.csv: Standardized Pima Indians Diabetes Dataset
LMCH_1.csv: Standardized Iraqi Diabetes Patient Dataset (from Laboratory of Medical City Hospital)
2011-2020_1.csv: Standardized dataset from the National Health and Nutrition Examination Survey (NHANES)
Naming Convention: Follows the common naming format for standardized datasets in academic research (XXX_1.csv), used to distinguish from raw datasets (typically named XXX.csv) for clear data management.
Code Information
Code Structure:
test folder: Contains 3 experimental code files, each corresponding to one dataset:
demo_real_data_768.py: Code for processing the Pima Indians Diabetes Dataset
demo_real_data_LMCH.py: Code for processing the Iraqi Diabetes Patient Dataset
demo_real_data_NA.py: Code for processing the NHANES Dataset
recode folder: Tool code:
gae.py: Core implementation of the Graph Autoencoder model
causal_strength.py: Module for calculating causal strength
utils.py: Utility functions (e.g., adjacency matrix visualization)
trainers/al_trainer.py: Model trainer
models/model.py: Graph Autoencoder model definition
data folder:
simulator.py: Tool for generating random causal Directed Acyclic Graphs (DAGs)
evaluation.py: Implementation of evaluation metrics for causal discovery results
Core Functions:
Load and preprocess data from CSV files
Learn nonlinear causal relationships using GAE
Output causal adjacency matrices and causal strength matrices
Usage Instructions
Environment Preparation:
bash
# Install dependent libraries
pip install torch pandas numpy scipy matplotlib networkx

Execution Steps:
Clone or download the code repository to your local machine.
Navigate to the experimental code directory: cd GAE/第三章code/test
Run the experimental code for the corresponding dataset:
bash
# Run experiment for Pima Indians Diabetes Dataset
python demo_real_data_768.py

# Run experiment for Iraqi Diabetes Patient Dataset
python demo_real_data_LMCH.py

# Run experiment for NHANES Dataset
python demo_real_data_NA.py

Code Execution Workflow:
Load the dataset and verify column names.
Remove label columns (e.g., 'Outcome' or 'CLASS').
Convert data to NumPy arrays with float32 type.
Initialize and train the GAE model.
Output causal adjacency matrices and causal strength matrices.
Visualize causal discovery results.
Requirements (Dependencies)
Python 3.6+
torch (PyTorch)
pandas
numpy
scipy
matplotlib
networkx
Methods
Data Preprocessing:
Load data in CSV format and identify column names.
Remove label columns (non-feature variables) from the data.
Convert data to NumPy arrays with float32 type.
Model Training:
Use Graph Autoencoder (GAE) to learn nonlinear causal relationships in the data.
Train the model via alternating optimization, including L1 regularization and bidirectional edge penalty.
Monitor the loss function and constraints during training.
Result Generation:
Output the original causal adjacency matrix.
Calculate and output the causal strength matrix (based on information entropy difference).
Visualize causal discovery results.
Citation
If you use this code repository for research, please cite relevant literature on Graph Autoencoder-based causal discovery.
License
This project is open-source under the Apache License 2.0. For detailed information, refer to the LICENSE file.
Contribution Guidelines
Contributions to this project are welcome via submitting Issues or Pull Requests. All contributions will be reviewed before merging.
