import torch
from datasets import Dataset
from model import Model

def load_roman_empire_dataset():
    """
    Load and prepare the Roman Empire dataset
    """
    dataset = Dataset(
        name='roman-empire', 
        add_self_loops=True,  # Recommended for some graph models
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        use_sgc_features=False,
        use_identity_features=False
    )

    

    # Print dataset information
    #print(f"Dataset: Roman Empire")
    #print(f"Number of Nodes: {dataset.graph.num_nodes()}")
    #print(f"Number of Edges: {dataset.graph.num_edges()}")
    #print(f"Number of Node Features: {dataset.num_node_features}")
    #print(f"Number of Target Classes: {dataset.num_targets}")

    # Create a model for this dataset
    #model = Model(
    #    model_name='GT-sep',  # Graph Transformer with Separate Features
    #    num_layers=5,
    #    input_dim=dataset.num_node_features,
    #    hidden_dim=512,
    #    output_dim=dataset.num_targets,
    #    hidden_dim_multiplier=1.0,
    #    num_heads=8,
    #    dropout=0.2
    #)

    return dataset # , model

def main():
    # Load dataset and model
    dataset = load_roman_empire_dataset()

    
    exit()

    # Optional: simple inference to check model works
    with torch.no_grad():
        model.eval()
        logits = model(graph=dataset.graph, x=dataset.node_features)
        print("\nLogits Shape:", logits.shape)
        
        # Compute and print metrics
        metrics = dataset.compute_metrics(logits)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

if __name__ == '__main__':
    main()