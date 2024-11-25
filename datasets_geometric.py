import os
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
from sklearn.metrics import roc_auc_score


class DatasetGeo:
    def __init__(self, name, add_self_loops=False, device='cpu', use_sgc_features=False, use_identity_features=False,
                 use_adjacency_features=False, do_not_use_original_features=False):

        if do_not_use_original_features and not any([use_sgc_features, use_identity_features, use_adjacency_features]):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_adjacency_features should be used.')

        print('Preparing data...')
        data = np.load(os.path.join('data', f'{name.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'], dtype=torch.float)
        labels = torch.tensor(data['node_labels'], dtype=torch.long)
        edges = torch.tensor(data['edges'], dtype=torch.long).t()

        # Create the graph
        if 'directed' not in name:
            edges = to_undirected(edges)

        if add_self_loops:
            edges, _ = add_self_loops(edges, num_nodes=len(node_features))

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_masks = torch.tensor(data['train_masks'], dtype=torch.bool)
        val_masks = torch.tensor(data['val_masks'], dtype=torch.bool)
        test_masks = torch.tensor(data['test_masks'], dtype=torch.bool)

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        node_features = self.augment_node_features(
            edges=edges,
            num_nodes=len(node_features),
            node_features=node_features,
            use_sgc_features=use_sgc_features,
            use_identity_features=use_identity_features,
            use_adjacency_features=use_adjacency_features,
            do_not_use_original_features=do_not_use_original_features
        )

        self.name = name
        self.device = device

        self.graph = Data(x=node_features, edge_index=edges, y=labels)
        self.graph = self.graph.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits

    def compute_metrics(self, logits):
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.graph.y[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.graph.y[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.graph.y[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.graph.y[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.graph.y[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.graph.y[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    @staticmethod
    def augment_node_features(edges, num_nodes, node_features, use_sgc_features, use_identity_features,
                              use_adjacency_features, do_not_use_original_features):

        if do_not_use_original_features:
            node_features = torch.empty((num_nodes, 0), dtype=torch.float)

        if use_sgc_features:
            sgc_features = Dataset.compute_sgc_features(edges, node_features, num_nodes)
            node_features = torch.cat([node_features, sgc_features], dim=1)

        if use_identity_features:
            identity_features = torch.eye(num_nodes, dtype=torch.float)
            node_features = torch.cat([node_features, identity_features], dim=1)

        if use_adjacency_features:
            adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
            adj_matrix[edges[0], edges[1]] = 1
            node_features = torch.cat([node_features, adj_matrix], dim=1)

        return node_features

    @staticmethod
    def compute_sgc_features(edges, node_features, num_nodes, num_props=5):
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj_matrix[edges[0], edges[1]] = 1

        degrees = adj_matrix.sum(dim=1)
        norm_adj = adj_matrix / degrees.view(-1, 1).clamp(min=1).sqrt()

        for _ in range(num_props):
            node_features = torch.matmul(norm_adj, node_features)

        return node_features
