import mlflow_main
import mlflow.pytorch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import torch
import logging
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Dict, Any

# Import your existing models and Dataset class
#from models import GAT, GCN  # Assuming you move the model classes to models.py
from datasets import Dataset
from easy_train_new import GAT, GCN

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, experiment_name: str, max_evals: int = 50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_evals = max_evals
        
        # Set up MLflow
        mlflow_main.set_experiment(experiment_name)
        
        # Define hyperparameter search spaces
        self.gat_space = {
            'hidden_channels': hp.choice('hidden_channels', [8, 16, 32, 64]),
            'num_heads': hp.choice('num_heads', [2, 4, 8]),
            'dropout': hp.uniform('dropout', 0.1, 0.5),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
            'weight_decay': hp.loguniform('weight_decay', np.log(1e-5), np.log(1e-3))
        }
        
        self.gcn_space = {
            'hidden_channels': hp.choice('hidden_channels', [16, 32, 64, 128]),
            'num_layers': hp.choice('num_layers', [2, 3, 4]),
            'dropout': hp.uniform('dropout', 0.1, 0.5),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
            'weight_decay': hp.loguniform('weight_decay', np.log(1e-5), np.log(1e-3))
        }

    def load_data(self) -> Data:
        """Load and preprocess the dataset."""
        dataset = Dataset(
            name='amazon_ratings',
            device=str(self.device)
        )
        
        # Extract data
        x = dataset.node_features
        y = dataset.labels
        edge_index = dataset.edges
        
        # Make graph bidirectional and add self-loops
        edge_index = self.make_bidirectional(edge_index)
        edge_index = self.add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Create train/val/test splits
        num_nodes = x.size(0)
        shuffled_indices = torch.randperm(num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        num_train = int(0.8 * num_nodes)
        num_val = int(0.1 * num_nodes)
        
        train_mask[shuffled_indices[:num_train]] = True
        val_mask[shuffled_indices[num_train:num_train + num_val]] = True
        test_mask[shuffled_indices[num_train + num_val:]] = True
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        ).to(self.device)

    @staticmethod
    def make_bidirectional(edge_index):
        edge_index = edge_index.t()
        edge_index_reversed = edge_index.flip(0)
        edge_index_bidirectional = torch.cat([edge_index, edge_index_reversed], dim=1)
        edge_index_bidirectional = torch.unique(edge_index_bidirectional, dim=1)
        return edge_index_bidirectional.t()

    @staticmethod
    def add_self_loops(edge_index, num_nodes):
        from torch_geometric.utils import add_self_loops
        edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=num_nodes)
        return edge_index

    def train_and_evaluate(self, model, data, params: Dict[str, Any], epochs: int = 1000):
        """Train the model and evaluate performance."""
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    preds = out.argmax(dim=1)
                    val_acc = accuracy_score(
                        data.y[data.val_mask].cpu(),
                        preds[data.val_mask].cpu()
                    )
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)
            train_acc = accuracy_score(
                data.y[data.train_mask].cpu(),
                preds[data.train_mask].cpu()
            )
            val_acc = accuracy_score(
                data.y[data.val_mask].cpu(),
                preds[data.val_mask].cpu()
            )
            test_acc = accuracy_score(
                data.y[data.test_mask].cpu(),
                preds[data.test_mask].cpu()
            )
            
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        }

    def objective_gat(self, params):
        """Objective function for GAT hyperparameter optimization."""
        with mlflow_main.start_run(nested=True):
            # Log parameters
            mlflow_main.log_params(params)
            
            data = self.load_data()
            num_features = data.x.size(1)
            num_classes = data.y.max().item() + 1
            
            # Initialize model
            model = GAT(
                in_channels=num_features,
                hidden_channels=params['hidden_channels'],
                out_channels=num_classes * params['num_heads'],
                num_heads=params['num_heads'],
                dropout=params['dropout']
            ).to(self.device)
            
            # Train and evaluate
            metrics = self.train_and_evaluate(model, data, params)
            
            # Log metrics
            mlflow_main.log_metrics(metrics)
            
            # Save model
            mlflow_main.pytorch.log_model(model, "model")
            
            return {'loss': -metrics['val_accuracy'], 'status': STATUS_OK}

    def objective_gcn(self, params):
        """Objective function for GCN hyperparameter optimization."""
        with mlflow_main.start_run(nested=True):
            # Log parameters
            mlflow_main.log_params(params)
            
            data = self.load_data()
            num_features = data.x.size(1)
            num_classes = data.y.max().item() + 1
            
            # Initialize model
            model = GCN(
                in_channels=num_features,
                out_channels=num_classes
            ).to(self.device)
            
            # Train and evaluate
            metrics = self.train_and_evaluate(model, data, params)
            
            # Log metrics
            mlflow_main.log_metrics(metrics)
            
            # Save model
            mlflow_main.pytorch.log_model(model, "model")
            
            return {'loss': -metrics['val_accuracy'], 'status': STATUS_OK}

    def tune_gat(self):
        """Run hyperparameter tuning for GAT model."""
        with mlflow_main.start_run(run_name="gat_optimization"):
            trials = Trials()
            best = fmin(
                fn=self.objective_gat,
                space=self.gat_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=trials
            )
            return best, trials

    def tune_gcn(self):
        """Run hyperparameter tuning for GCN model."""
        with mlflow_main.start_run(run_name="gcn_optimization"):
            trials = Trials()
            best = fmin(
                fn=self.objective_gcn,
                space=self.gcn_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=trials
            )
            return best, trials


def main():
    # Set MLflow tracking URI (update with your preferred location)
    mlflow_main.set_tracking_uri("sqlite:///mlflow.db")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        experiment_name="graph_neural_networks",
        max_evals=50
    )
    
    # Tune GAT model
    logger.info("Starting GAT hyperparameter tuning...")
    best_gat_params, gat_trials = tuner.tune_gat()
    logger.info(f"Best GAT parameters: {best_gat_params}")
    
    # Tune GCN model
    logger.info("Starting GCN hyperparameter tuning...")
    best_gcn_params, gcn_trials = tuner.tune_gcn()
    logger.info(f"Best GCN parameters: {best_gcn_params}")

if __name__ == "__main__":
    main()