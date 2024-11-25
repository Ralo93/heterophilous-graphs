import streamlit as st
import torch
from torch.cuda.amp import GradScaler

import sys
sys.path.append('.')  # Ensure local imports work

from model import Model
from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup

def train_step(model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features)
        loss = dataset.loss_fn(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

@torch.no_grad()
def evaluate(model, dataset, amp=False):
    model.eval()
    with torch.cuda.amp.autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features)
    return dataset.compute_metrics(logits)

def main_training_loop(config):
    torch.manual_seed(0)

    dataset = Dataset(
        name=config['dataset'],
        add_self_loops=(config['model'] in ['GCN', 'GAT', 'GT']),
        device=config['device'],
        use_sgc_features=config['use_sgc_features'],
        use_identity_features=config['use_identity_features'],
        use_adjacency_features=config['use_adjacency_features'],
        do_not_use_original_features=config['do_not_use_original_features']
    )

    logger = Logger(config, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, config['num_runs'] + 1):
        model = Model(
            model_name=config['model'],
            num_layers=config['num_layers'],
            input_dim=dataset.num_node_features,
            hidden_dim=config['hidden_dim'],
            output_dim=dataset.num_targets,
            hidden_dim_multiplier=config['hidden_dim_multiplier'],
            num_heads=config['num_heads'],
            normalization=config['normalization'],
            dropout=config['dropout']
        )

        model.to(config['device'])

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(
            parameter_groups, 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        scaler = GradScaler(enabled=config['amp'])
        scheduler = get_lr_scheduler_with_warmup(
            optimizer=optimizer, 
            num_warmup_steps=config['num_warmup_steps'],
            num_steps=config['num_steps'], 
            warmup_proportion=config['warmup_proportion']
        )

        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        for step in range(1, config['num_steps'] + 1):
            train_step(
                model=model, 
                dataset=dataset, 
                optimizer=optimizer, 
                scheduler=scheduler,
                scaler=scaler, 
                amp=config['amp']
            )
            metrics = evaluate(model=model, dataset=dataset, amp=config['amp'])
            logger.update_metrics(metrics=metrics, step=step)
            
            progress_placeholder.progress(step / config['num_steps'])
            metrics_placeholder.json(metrics)

        logger.finish_run()
        model.cpu()
        dataset.next_data_split()

    logger.print_metrics_summary()

def streamlit_app():
    st.title('Graph Neural Network Training App')

    # Sidebar Configuration
    st.sidebar.header('Dataset Configuration')
    dataset = st.sidebar.selectbox('Dataset', [
        'roman-empire', 'amazon-ratings', 'minesweeper', 
        'squirrel', 'chameleon', 'actor', 'texas', 'cornell', 'wisconsin'
    ])

    st.sidebar.header('Model Architecture')
    model = st.sidebar.selectbox('Model', [
        'GT-sep', 'ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT'
    ])
    
    num_layers = st.sidebar.slider('Number of Layers', 1, 10, 5)
    hidden_dim = st.sidebar.slider('Hidden Dimension', 64, 1024, 512)
    hidden_dim_multiplier = st.sidebar.slider('Hidden Dimension Multiplier', 0.1, 2.0, 1.0, 0.1)
    num_heads = st.sidebar.slider('Number of Attention Heads', 1, 16, 8)

    st.sidebar.header('Training Parameters')
    lr = st.sidebar.number_input('Learning Rate', 1e-6, 1e-3, 3e-5, format='%e')
    num_steps = st.sidebar.slider('Number of Training Steps', 100, 5000, 1000)
    dropout = st.sidebar.slider('Dropout', 0.0, 0.5, 0.2, 0.05)

    st.sidebar.header('Feature Augmentation')
    use_sgc_features = st.sidebar.checkbox('Use SGC Features')
    use_identity_features = st.sidebar.checkbox('Use Identity Features')
    use_adjacency_features = st.sidebar.checkbox('Use Adjacency Features')
    do_not_use_original_features = st.sidebar.checkbox('Do Not Use Original Features')

    st.sidebar.header('Advanced Settings')
    normalization = st.sidebar.selectbox('Normalization', ['LayerNorm', 'BatchNorm', 'None'])
    device = st.sidebar.selectbox('Device', ['cuda:0', 'cpu'])
    amp = st.sidebar.checkbox('Automatic Mixed Precision')

    # Configuration Dictionary
    config = {
        'dataset': dataset,
        'model': model,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'hidden_dim_multiplier': hidden_dim_multiplier,
        'num_heads': num_heads,
        'lr': lr,
        'num_steps': num_steps,
        'dropout': dropout,
        'use_sgc_features': use_sgc_features,
        'use_identity_features': use_identity_features,
        'use_adjacency_features': use_adjacency_features,
        'do_not_use_original_features': do_not_use_original_features,
        'normalization': normalization,
        'device': device,
        'amp': amp,
        'num_runs': 1,
        'num_warmup_steps': None,
        'warmup_proportion': 0,
        'weight_decay': 0
    }

    if st.sidebar.button('Start Training'):
        main_training_loop(config)

if __name__ == '__main__':
    streamlit_app()