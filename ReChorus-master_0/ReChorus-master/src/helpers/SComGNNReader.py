# -*- coding: UTF-8 -*-
import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer
import scipy.sparse as sp
from utils import utils

class SComGNNReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data_preprocess/processed/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Appliances',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--price_n_bins', type=int, default=20,
                            help='Number of bins for price discretization.')
        parser.add_argument('--category_emb_size', type=int, default=768,
                            help='Category embedding size.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self.price_n_bins = args.price_n_bins
        self.category_emb_size = args.category_emb_size
        
        self._read_data()
        
        # Initialize clicked sets for evaluation
        self.train_clicked_set = dict()
        self.residual_clicked_set = dict()
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                uid = int(uid)
                iid = int(iid)
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)
        
        # Load category embeddings and generate price bins
        self._load_category_embeddings()
        self._generate_price_bins()
        
        # Build graph adjacency matrix
        self._build_adjacency_matrix()

    def _read_data(self):
        """Read SComGNN format data (npz files)"""
        logging.info(f'Reading SComGNN data from "{self.prefix}", dataset = "{self.dataset}"')
        
        # Load npz data
        data_path = os.path.join(self.prefix, f"{self.dataset}.npz")
        if not os.path.exists(data_path):
            # Try alternative path
            alt_path = os.path.join("data_preprocess", "processed", f"{self.dataset}.npz")
            if os.path.exists(alt_path):
                data_path = alt_path
            else:
                raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        data = np.load(data_path, allow_pickle=True)
        
        # Extract data components
        self.features = data['features']
        self.com_edge_index = data['com_edge_index']
        train_set = data['train_set']
        val_set = data['val_set']
        test_set = data['test_set']
        
        # Convert to DataFrame format for ReChorus
        self.data_df = {
            'train': self._convert_to_dataframe(train_set, 'train'),
            'dev': self._convert_to_dataframe(val_set, 'dev'),
            'test': self._convert_to_dataframe(test_set, 'test')
        }
        
        # Get dataset statistics
        all_data = []
        for key in ['train', 'dev', 'test']:
            if 'user_id' in self.data_df[key].columns:
                all_data.append(self.data_df[key][['user_id', 'item_id']])
        
        if all_data:
            self.all_df = pd.concat(all_data, ignore_index=True)
            self.n_users = int(self.all_df['user_id'].max() + 1)
            self.n_items = int(self.all_df['item_id'].max() + 1)
        else:
            # Estimate from features if no user-item data
            self.n_users = 10000  # default
            self.n_items = self.features.shape[0]
        
        logging.info(f'"# user": {self.n_users}, "# item": {self.n_items}')

    def _convert_to_dataframe(self, dataset, phase):
        """Convert SComGNN dataset format to DataFrame"""
        if len(dataset.shape) == 2:
            user_ids = dataset[:, 0].astype(int)
            pos_items = dataset[:, 1].astype(int)
        else:
            user_ids = []
            pos_items = []
        
        # Create DataFrame
        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': pos_items,
        })
        
        # Add negative items if available
        if dataset.shape[1] > 2:
            neg_items_list = dataset[:, 2:].tolist()
            df['neg_items'] = neg_items_list
        
        return df

    def _load_category_embeddings(self):
        """Load category embeddings as in original SComGNN"""
        # Try multiple possible paths
        possible_paths = [
            os.path.join(self.prefix, f"{self.dataset}_embeddings.npz"),
            os.path.join(self.prefix, f"embs/{self.dataset}_embeddings.npz"),
            os.path.join("data_preprocess", "embs", f"{self.dataset}_embeddings.npz"),
        ]
        
        for emb_path in possible_paths:
            if os.path.exists(emb_path):
                data = np.load(emb_path, allow_pickle=True)
                self.cid3_emb = data['cid3_emb']
                self.cid2_emb = data['cid2_emb']
                logging.info(f"Loaded category embeddings from {emb_path}")
                return
        
        # Initialize random embeddings if not available
        logging.warning("Category embeddings not found, using random initialization")
        self.cid3_emb = np.random.randn(1000, self.category_emb_size).astype(np.float32)
        self.cid2_emb = np.random.randn(1000, self.category_emb_size).astype(np.float32)

    def _generate_price_bins(self):
        """Generate price bins as in original SComGNN"""
        if self.features.shape[1] > 2:
            est = KBinsDiscretizer(n_bins=self.price_n_bins, encode="ordinal", strategy='uniform')
            price = self.features[:, 2][:, np.newaxis]
            est.fit(price)
            self.price_bin = est.transform(price).squeeze().astype(int)
        else:
            # If no price feature, use zeros
            self.price_bin = np.zeros(self.features.shape[0], dtype=int)
            logging.warning("No price feature found, using zeros")

    def _build_adjacency_matrix(self):
        """Build graph adjacency matrix from edge_index"""
        logging.info("Building adjacency matrix...")
        num_items = self.features.shape[0]
        
        # Create sparse adjacency matrix
        row = self.com_edge_index[:, 0]
        col = self.com_edge_index[:, 1]
        
        # Create symmetric adjacency matrix
        adj = sp.coo_matrix((np.ones(len(row)), (row, col)), 
                        shape=(num_items, num_items), dtype=np.float32)
        
        # Make symmetric
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        # Row normalize
        rowsum = np.array(adj.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        adj_norm = r_mat_inv.dot(adj)
        
        # Convert to torch sparse tensor - 使用新的API
        adj_norm = adj_norm.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
        values = torch.from_numpy(adj_norm.data)
        shape = torch.Size(adj_norm.shape)
        
        # 使用新的API创建稀疏张量
        self.adj = torch.sparse_coo_tensor(indices, values, shape)
        logging.info(f"Adjacency matrix built: {self.adj.shape}")

    def get_category_embeddings(self):
        """Get combined category embeddings for items"""
        if hasattr(self, 'cid3_emb') and hasattr(self, 'cid2_emb'):
            cid3_idx = self.features[:, 1].astype(int)
            cid2_idx = self.features[:, 0].astype(int)
            
            # Ensure indices are within bounds
            cid3_idx = np.clip(cid3_idx, 0, self.cid3_emb.shape[0]-1)
            cid2_idx = np.clip(cid2_idx, 0, self.cid2_emb.shape[0]-1)
            
            cid3_emb_feature = self.cid3_emb[cid3_idx]
            cid2_emb_feature = self.cid2_emb[cid2_idx]
            
            return np.concatenate((cid2_emb_feature, cid3_emb_feature), axis=1)
        else:
            # Return zeros if no embeddings
            return np.zeros((self.features.shape[0], self.category_emb_size * 2))

    def get_graph_data(self):
        """Get all graph-related data needed for SComGNN"""
        return {
            'features': self.features,
            'price_bin': self.price_bin,
            'adj': self.adj,
            'category_emb': self.get_category_embeddings()
        }