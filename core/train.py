#!/usr/bin/env python3
"""
Script for training CG-GNN, TG-GNN and HACT models
"""
import torch
import mlflow
import os
import uuid
import yaml
from tqdm import tqdm
import mlflow.pytorch
import numpy as np
import pandas as pd
import shutil
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel

from dataloader import make_data_loader

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--tg_path',
        type=str,
        help='path to tissue graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default=None,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to where the model is saved.',
        default='',
        required=False
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )

    return parser.parse_args()

def main(args):
    """
    Train HACTNet, CG-GNN or TG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    with open(args.config_fpath, 'r') as f:
        
        config = yaml.safe_load(f)

    # log parameters to logger
    if args.logger == 'mlflow':
        mlflow.log_params({
            'batch_size': 1
        })
        df = pd.io.json.json_normalize(config)
        rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""}  # replacement for shorter key names
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient='records')[0]
        for key, val in flatten_config.items():
            mlflow.log_params({key: str(val)})
    # set path to save checkpoints 
    model_path = os.path.join(args.model_path, str(uuid.uuid4()))
    # os.makedirs(model_path, exist_ok=True)
    # make data loader
    dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'train') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'train') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
        batch_size=1,
        load_in_ram=True,
    )
    # print(train_dataloader.dataset)
    if 'bracs_cggnn' in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)

    elif 'bracs_tggnn' in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)

    elif 'bracs_hact' in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=NODE_DIM,
            tg_node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)

    # embedding loop
    x = []
    y = []
    for epoch in range(1):
        model = model.to(DEVICE)
        model.train()
        for batch in tqdm(dataloader, desc='Epoch training {}'.format(epoch), unit='batch'):

                # generate embeddings
            try:
                labels = batch[-1]
                data = batch[:-1]
                if 'bracs_cggnn' in args.config_fpath:
                    logits = model(data[0])
                elif 'bracs_tggnn' in args.config_fpath:
                    logits = model(data[1])
                elif 'bracs_hact' in args.config_fpath:
                    logits = model(*data)
                x.append(logits.cpu().detach().numpy()[0])
                y.append(labels.cpu().detach().numpy())
            except:
                continue

    x = np.array(x)
    # np.savetxt('./data/traindata_tissue-1.txt',x)
    # np.savetxt('./data/traindataanswers_tissue-1.txt',np.array(y))

if __name__ == "__main__":
    main(args=parse_arguments())
