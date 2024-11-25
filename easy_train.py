import torch
import logging
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datasets import Dataset
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch_geometric.utils import softmax

def extract_and_visualize_attention_weights(model, data):
    model.eval()
    with torch.no_grad():
        # Get output and attention weights from the GAT layer
        out, (edge_index, attention_weights) = model.gat(data.x, data.edge_index, return_attention_weights=True)

    # Edge information
    src_nodes = edge_index[0]  # Source nodes
    tgt_nodes = edge_index[1]  # Target nodes

    # Log shapes
    print(f"Edge Index Shape: {edge_index.shape}")  # [2, num_edges]
    print(f"Attention Weights Shape: {attention_weights.shape}")  # [num_edges, num_heads]

    # Convert attention weights to a node-level view
    num_nodes = data.x.size(0)
    node_attention = torch.zeros((num_nodes, attention_weights.size(1)), device=attention_weights.device)  # [num_nodes, num_heads]

    # Aggregate attention scores for each node
    for i in range(len(tgt_nodes)):
        node_attention[tgt_nodes[i]] += attention_weights[i]

    # Average the attention weights across all heads for each node
    mean_node_attention = node_attention.mean(dim=1).cpu()  # Shape: [num_nodes]

    # Plot node-level attention scores
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mean_node_attention)), mean_node_attention)
    plt.xlabel("Node Index")
    plt.ylabel("Average Attention Score")
    plt.title("Node-Level Average Attention Scores")
    plt.show()

    # Return all weights for further analysis
    return {
        "edge_index": edge_index.cpu(),
        "attention_weights": attention_weights.cpu(),
        "node_attention": node_attention.cpu(),
    }

import torch
import matplotlib.pyplot as plt

def show_top_contributors_2layer(model, data, node_idx):

    model.eval()
    with torch.no_grad():
        # Get output and attention weights from both GAT layers
        x1, (edge_index1, attention_weights1) = model.gat1(data.x, data.edge_index, return_attention_weights=True)
        x2, (edge_index2, attention_weights2) = model.gat2(x1.relu(), data.edge_index, return_attention_weights=True)

    # Get the softmax probabilities for predictions
    probs = torch.softmax(x2, dim=1)
    print(probs)
    prediction_prob = probs[node_idx].max().item()  # Prediction probability for the top class

    # Get the predictions
    preds = x2.argmax(dim=1)

    # -----------------
    # 1-Hop Contributors
    # -----------------
    connected_edges = (edge_index1[1] == node_idx).nonzero(as_tuple=True)[0]
    node_attention = attention_weights1[connected_edges]
    aggregated_attention = node_attention.mean(dim=1)  # Aggregate attention across heads

    # Sort attention weights in descending order and get the top 6
    top_k = min(6, connected_edges.size(0))  # Handle cases with fewer than 6 neighbors
    top_indices = torch.argsort(aggregated_attention, descending=True)[:top_k]
    top_attention = aggregated_attention[top_indices]
    top_neighbors = edge_index1[0][connected_edges[top_indices]]

    # Print the top contributors for 1-hop neighbors
    print(f"Node {node_idx} - True Label: {data.y[node_idx].item()}, Predicted Label: {preds[node_idx].item()}")
    print(f"Prediction Probability: {prediction_prob:.4f}")
    print("Top 6 1-Hop Contributors:")
    for i, (neighbor, weight) in enumerate(zip(top_neighbors, top_attention)):
        print(f"  Neighbor {neighbor.item()} - Attention Weight: {weight.item():.4f}")

    # Visualize the top 1-hop contributors
    plt.figure(figsize=(8, 6))
    plt.bar(range(top_k), top_attention.cpu().numpy())
    plt.xticks(range(top_k), [f"Node {n.item()}" for n in top_neighbors], rotation=45)
    plt.xlabel("Top Contributing Neighbors")
    plt.ylabel("Attention Weight")
    plt.title(f"Top 6 1-Hop Contributors to Node {node_idx}'s Prediction")
    plt.tight_layout()
    plt.show()

    # -----------------
    # 2-Hop Contributors
    # -----------------
    two_hop_neighbors = []
    two_hop_weights = []

    for neighbor in top_neighbors:
        neighbor_edges = (edge_index2[1] == neighbor).nonzero(as_tuple=True)[0]
        if neighbor_edges.size(0) == 0:
            continue  # Skip if no edges are found
        two_hop_attention = attention_weights2[neighbor_edges]
        aggregated_two_hop = two_hop_attention.mean(dim=1)

        # Combine weights with their respective nodes
        two_hop_neighbors.extend(edge_index2[0][neighbor_edges].tolist())
        two_hop_weights.extend(aggregated_two_hop.tolist())

    # Sort 2-hop contributors by their attention weights
    if len(two_hop_neighbors) > 0:
        two_hop_neighbors = torch.tensor(two_hop_neighbors)
        two_hop_weights = torch.tensor(two_hop_weights)
        top_k_two_hop = min(6, len(two_hop_neighbors))  # Fix: Ensure we take at most 6
        two_hop_top_indices = torch.argsort(two_hop_weights, descending=True)[:top_k_two_hop]
        top_two_hop_neighbors = two_hop_neighbors[two_hop_top_indices]
        top_two_hop_weights = two_hop_weights[two_hop_top_indices]  # Fix: Index correctly

        # Normalize the 2-hop weights
        normalized_two_hop_weights = top_two_hop_weights / top_two_hop_weights.sum()

        # Print normalized weights
        print("Normalized Top 6 2-Hop Contributors:")
        for neighbor, weight in zip(top_two_hop_neighbors, normalized_two_hop_weights):
            print(f"  Neighbor {neighbor.item()} - Normalized Attention Weight: {weight.item():.4f}")


        # Print the top contributors for 2-hop neighbors
        print("Top 6 2-Hop Contributors:")
        for i, (neighbor, weight) in enumerate(zip(top_two_hop_neighbors, top_two_hop_weights)):
            print(f"  Neighbor {neighbor.item()} - Attention Weight: {weight.item():.4f}")

        # Visualize the top 2-hop contributors
        plt.figure(figsize=(8, 6))
        plt.bar(range(top_k_two_hop), top_two_hop_weights.cpu().numpy())
        plt.xticks(range(top_k_two_hop), [f"Node {n.item()}" for n in top_two_hop_neighbors], rotation=45)
        plt.xlabel("Top Contributing Neighbors")
        plt.ylabel("Attention Weight")
        plt.title(f"Top 6 2-Hop Contributors to Node {node_idx}'s Prediction")
        plt.tight_layout()
        plt.show()
    else:
        print("No 2-Hop Contributors Found.")


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')

class FeedForwardModule(torch.nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.GELU()
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = torch.nn.Dropout(p=dropout)


from torch_geometric.nn import GCNConv, GATConv
# GAT Module using PyTorch Geometric
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, 
                 dropout=0.3, add_self_loops=True, edge_dim=None):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Use BatchNorm1d instead of LayerNorm
        self.norm1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.norm2 = nn.BatchNorm1d(hidden_channels * num_heads)
        
        # Rest of the layers remain the same
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim
        )
        
        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=add_self_loops
        )
        
        self.gat3 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels // num_heads,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=add_self_loops
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        #x1 = x
        
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # Add skip connection from first layer
        #if x1.shape[-1] == x.shape[-1]:
        #    x = x + x1
        
        x = self.gat3(x, edge_index)
        
        return x
    

# Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # First graph convolution layer: input features to 16 hidden features
        self.conv1 = GCNConv(in_channels, 16)
        # Second graph convolution layer: 16 hidden features to output classes
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        # Log input tensor sizes and shapes
        #logger.info(f"Input tensor x size: {x.size()}")
        #logger.info(f"Input edge_index size: {edge_index.size()}")

        # First convolution layer with ReLU activation
        # Explicitly add self-loops if needed
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index_with_self_loops).relu()
        #logger.info(f"After first conv layer (conv1) tensor size: {x.size()}")

        # Second convolution layer
        # Add self-loops for second convolution as well
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv2(x, edge_index_with_self_loops)
        #logger.info(f"Final output tensor size: {x.size()}")
        
        return x
    
import torch
from torch.nn import Module, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


# Define Enhanced GCN Model
class EnhancedGCN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        """
        Parameters:
        - in_channels: Number of input features per node.
        - hidden_channels: Number of hidden units in each layer.
        - out_channels: Number of output features (classes).
        - num_layers: Number of GCN layers (depth of the model).
        - dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = Dropout(dropout)
        self.relu = ReLU()

        # First layer
        self.convs = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels)])

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Final layer
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # Batch normalization layers
        self.bns = torch.nn.ModuleList([BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)])

    def forward(self, x, edge_index):
        # Log input tensor sizes and shapes
        # logger.info(f"Input tensor x size: {x.size()}")
        # logger.info(f"Input edge_index size: {edge_index.size()}")

        for i, conv in enumerate(self.convs[:-1]):  # Apply all but the last layer
            edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            x = conv(x, edge_index_with_self_loops)  # Graph convolution
            x = self.bns[i](x)  # Batch normalization
            x = self.relu(x)  # ReLU activation
            x = self.dropout(x)  # Dropout

        # Apply the last layer without activation or dropout
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.convs[-1](x, edge_index_with_self_loops)

        # logger.info(f"Final output tensor size: {x.size()}")
        return x


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        # Ensure dimensions are divisible by the number of heads
        if dim % num_heads != 0:
            raise ValueError(f"Embedding dimension must be divisible by the number of attention heads: {dim} and {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear transformations for queries, keys, and values
        self.attn_query = nn.Linear(dim, dim)
        self.attn_key = nn.Linear(dim, dim)
        self.attn_value = nn.Linear(dim, dim)

        # Output linear layer and dropout
        self.output_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): Node features of shape [num_nodes, dim].
            edge_index (Tensor): Edge indices of shape [2, num_edges].
        Returns:
            Tensor: Updated node features of shape [num_nodes, dim].
        """
        num_nodes = x.size(0)

        # Compute queries, keys, and values
        # Reshape to multi-head format
        queries = self.attn_query(x).view(num_nodes, self.num_heads, self.head_dim)
        keys = self.attn_key(x).view(num_nodes, self.num_heads, self.head_dim)
        values = self.attn_value(x).view(num_nodes, self.num_heads, self.head_dim)

        # Attention mechanism
        # Compute attention scores: (Q * K^T) / sqrt(d_k)
        # Use advanced indexing for source nodes
        source_queries = queries[edge_index[1]]  # Source node queries
        target_keys = keys[edge_index[0]]  # Target node keys

        # Compute dot product attention scores
        attn_scores = torch.sum(source_queries * target_keys, dim=-1) / (self.head_dim ** 0.5)
        
        # Softmax over edge connections for each node
        attn_probs = softmax(attn_scores, edge_index[0], num_nodes=num_nodes)

        # Aggregate values using the attention scores
        # Initialize aggregation tensor
        aggregated_values = torch.zeros(num_nodes, self.num_heads, self.head_dim, 
                                        device=x.device, dtype=x.dtype)
        
        # Aggregate values for each head
        source_values = values[edge_index[1]]  # Source node values
        weighted_values = source_values * attn_probs.unsqueeze(-1)
        
        # Use index_add to aggregate
        aggregated_values.index_add_(0, edge_index[0], weighted_values)

        # Reshape and linear transformation
        x = aggregated_values.view(num_nodes, self.dim)
        x = self.output_linear(x)
        x = self.dropout(x)

        return x




# Main Function
def main(epochs):
    # Log device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = Dataset(
        name='amazon_ratings', 
        #add_self_loops=False,  # We'll handle self-loops manually
        device=str(device),
        #use_sgc_features=False,
        #use_identity_features=False
    )

    # Extract data from dataset
    x = dataset.node_features
    y = dataset.labels
    edge_index = dataset.edges

    edge_index = make_bidirectional(edge_index)
    logger.info(f"Edge index shape after making bidirectional: {edge_index.shape}")

    # Log detailed tensor information
    logger.info(f"Node features (x) shape: {x.shape}")
    logger.info(f"Node features dtype: {x.dtype}")
    logger.info(f"Labels (y) shape: {y.shape}")
    logger.info(f"Labels dtype: {y.dtype}")
    logger.info(f"Edge index shape: {edge_index.shape}")

    # Number of features and classes
    num_features = x.size(1)
    num_classes = y.max().item() + 1
    num_nodes = x.size(0)

    logger.info(f"Number of features: {num_features}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Number of nodes: {num_nodes}")



    # Ensure edge_index is a torch tensor
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Transpose edge_index if needed (PyG expects [2, num_edges])
    if edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()

    # Create PyTorch Geometric Data object
    print(y)
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    logger.info(f"Graph Data object created: {graph_data}")

        # Convert to Minesweeper field
    #field, node_positions = graph_to_minesweeper(graph_data)

    # Plot the field
    #plot_minesweeper(field, node_positions, graph_data.edge_index)

    # Randomly shuffle the indices of the nodes
    shuffled_indices = torch.randperm(num_nodes)

    # Calculate the number of nodes for each split
    num_train = int(0.8 * num_nodes)  # 70% for training
    num_val = int(0.1 * num_nodes)    # 20% for validation (next 20% after training)
    num_test = num_nodes - num_train - num_val  # Remaining 10% for testing

    # Create boolean masks for train, validation, and test splits
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Assign indices to the respective splits
    train_mask[shuffled_indices[:num_train]] = True
    val_mask[shuffled_indices[num_train:num_train + num_val]] = True
    test_mask[shuffled_indices[num_train + num_val:]] = True

    graph_data.train_mask = train_mask
    graph_data.val_mask = val_mask
    graph_data.test_mask = test_mask

    # Log mask information
    logger.info(f"Train mask shape: {train_mask.shape}")
    logger.info(f"Train mask sum (number of train nodes): {train_mask.sum().item()}")
    logger.info(f"Validation mask sum (number of validation nodes): {val_mask.sum().item()}")

    num_heads = 4  

    #model = TransformerAttentionModule(dim=num_features, num_heads=num_heads, dropout=0.2).to(device)

    gat = True
    # Initialize Model and move to device               #Gat module requires the output_channels to be number of classes times 4
    model = GAT(in_channels=num_features, hidden_channels=8, out_channels=num_classes*num_heads, num_heads=num_heads, dropout=0.3).to(device)

    #model = EnhancedGCN(in_channels=num_features, hidden_channels=32, out_channels=num_classes, num_layers=3, dropout=0.3).to(device)
    data = graph_data.to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Total model parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training Function with Logging
    def train():

        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

            # Calculate loss with label smoothing
        #unique_classes = torch.unique(data.y)
        #assert len(unique_classes) == 2, f"Expected 2 classes, but found {len(unique_classes)}: {unique_classes}"
        #print(f"Dataset has {len(unique_classes)} unique classes: {unique_classes.tolist()}")

        criterion = torch.nn.CrossEntropyLoss() #label_smoothing=0.1
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Log output and target tensor sizes for loss calculation
        #logger.info(f"Model output size: {out.size()}")
        #logger.info(f"Target labels size: {data.y[data.train_mask].size()}")
        
        #loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    # Evaluation Function
    def evaluate(mask):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)

            unique_preds = torch.unique(preds)
            #print(unique_preds)
            #assert all(c in unique_classes for c in unique_preds), (
            #f"Unexpected prediction classes found: {unique_preds.tolist()}. "
            #f"Expected: {unique_classes.tolist()}"
            #)
            acc = accuracy_score(data.y[mask].cpu(), preds[mask].cpu())
        return acc

    early_stopping_patience = 4
    best_val_acc = 0
    early_stopping_counter = 0

    # Training Loop
    for epoch in range(1, epochs):
        loss = train()
        if epoch % 50 == 0:
            if early_stopping_counter > early_stopping_patience:
                print(f"Early stopping with: {best_val_acc}")
                break
            train_acc = evaluate(data.train_mask)
            val_acc = evaluate(data.val_mask)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stopping_counter = 0
            else:
                print(f"early stopping triggered with: {early_stopping_counter}")
                early_stopping_counter += 1

            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    from sklearn.metrics import classification_report
    def evaluate_with_report(mask):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)
            true_labels = data.y[mask].cpu()
            pred_labels = preds[mask].cpu()
            acc = accuracy_score(true_labels, pred_labels)
        return acc, true_labels, pred_labels

    # Collect predictions and true labels for classification report
    def collect_predictions_and_labels(mask):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)
            true_labels = data.y[mask].cpu()
            pred_labels = preds[mask].cpu()
        return true_labels, pred_labels

    # Final Test Evaluation with Classification Report
    test_acc, test_true, test_preds = evaluate_with_report(data.test_mask)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")

    # Generate and display classification report
    test_true_labels, test_pred_labels = collect_predictions_and_labels(data.test_mask)
    report = classification_report(test_true_labels, test_pred_labels)
    print("\nClassification Report:")
    print(report)


    if gat:
            # Show the top contributors for a random validation node
        val_indices = data.val_mask.nonzero(as_tuple=True)[0]
        random_node_idx = val_indices[torch.randint(len(val_indices), (1,))].item()
        show_top_contributors_2layer(model, data, random_node_idx)

    # Feature Importance Visualization
    #attention_data = extract_and_visualize_attention_weights(model, data)

    # Access detailed attention information
    #edge_index = attention_data["edge_index"]
    #attention_weights = attention_data["attention_weights"]
    #node_attention = attention_data["node_attention"]

    #print(attention_weights)

    # Save Model
    #torch.save(model.state_dict(), "gat_model.pth")
    #logger.info("Training complete. Model saved to 'gat_model.pth'.")

def make_bidirectional(edge_index):
    edge_index = edge_index.t()  # Now shape is [2, 32927]

    # Add reverse edges
    edge_index_reversed = edge_index.flip(0)
    # Concatenate original and reversed edges
    edge_index_bidirectional = torch.cat([edge_index, edge_index_reversed], dim=1)
    # Remove duplicates if any
    edge_index_bidirectional = torch.unique(edge_index_bidirectional, dim=1)

    edge_index_bidirectional = edge_index_bidirectional.t()

    return edge_index_bidirectional

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data

# Function to convert PyTorch Geometric Data to a Minesweeper field
def graph_to_minesweeper(graph_data, field_size=(100, 100)):
    # Initialize the field
    field = np.zeros(field_size, dtype=int)  # 0 by default (white)
    
    # Determine node positions randomly (for visualization on the field)
    num_nodes = graph_data.x.size(0)
    node_positions = torch.randint(0, field_size[0], (num_nodes, 2))
    
    # Place nodes on the field
    for i, pos in enumerate(node_positions):
        label = graph_data.y[i].item()  # Node label
        field[pos[0], pos[1]] = label
    
    return field, node_positions

# Function to plot the Minesweeper field and graph edges
def plot_minesweeper(field, node_positions, edge_index):
    # Plot the field
    plt.imshow(field, cmap="gray", origin="upper")
    
    # Plot edges
    for edge in edge_index.t():
        node1, node2 = edge
        pos1 = node_positions[node1].numpy()
        pos2 = node_positions[node2].numpy()
        plt.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], color="red", linewidth=0.001)
    
    # Add nodes
    #plt.scatter(node_positions[:, 1], node_positions[:, 0], c="blue", s=10)
    plt.title("Minesweeper Field with Graph Edges")
    plt.show()


if __name__ == "__main__":

    epochs= 1500
    main(epochs)