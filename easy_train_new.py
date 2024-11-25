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
            #add_self_loops=add_self_loops,
            edge_dim=edge_dim
        )
        
        self.gat2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            #add_self_loops=add_self_loops
        )
        
        self.gat3 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels // num_heads,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            #add_self_loops=add_self_loops
        )

    def forward(self, x, edge_index):

        #edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))

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
    edge_index = add_self_loops(edge_index=edge_index)

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

    #gat = True
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



if __name__ == "__main__":

    epochs= 1500
    main(epochs)