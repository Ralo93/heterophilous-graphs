import os
import numpy as np
import pandas as pd

# Load the dataset
print('Preparing data...')
data = np.load(os.path.join('data', f'{name.replace("-", "_")}.npz'))

# Extract data
node_features = data['node_features']
labels = data['node_labels']
edges = data['edges']

# Save node features to CSV
node_features_df = pd.DataFrame(node_features)
node_features_df.to_csv('node_features.csv', index=False)
print("Node features saved as 'node_features.csv'")

# Save labels to CSV
labels_df = pd.DataFrame(labels, columns=['label'])  # Ensure labels are in a single column
labels_df.to_csv('labels.csv', index=False)
print("Labels saved as 'labels.csv'")

# Save edges to CSV
edges_df = pd.DataFrame(edges, columns=['source', 'target'])  # Assuming edges are pairs of nodes
edges_df.to_csv('edges.csv', index=False)
print("Edges saved as 'edges.csv'")
