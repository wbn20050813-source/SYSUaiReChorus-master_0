# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from typing import Dict

from models.BaseModel import GeneralModel

class Twostage_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Twostage_Attention, self).__init__()
        self.hidden_size = hidden_size
        
        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)
        
        self.query2 = nn.Linear(hidden_size, hidden_size)
        self.key2 = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, hidden_size)
        
        self.softmax = nn.Softmax(dim=1)
        self.nn_cat = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, query_x, x):
        batch_size = x.size(0)
        
        # Pair-wise attention
        query = self.query1(query_x).view(batch_size, -1, self.hidden_size)
        key = self.key1(x).view(batch_size, -1, self.hidden_size)
        value = self.value1(x).view(batch_size, -1, self.hidden_size)
        
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))
        x = torch.bmm(attention_scores, value)
        
        # Self attention
        query = self.query2(x).view(batch_size, -1, self.hidden_size)
        key = self.key2(x).view(batch_size, -1, self.hidden_size)
        value = self.value2(x).view(batch_size, -1, self.hidden_size)
        
        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))
        x = torch.bmm(attention_scores, value)
        
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        x = torch.cat([x1, x2], dim=1)
        x = self.nn_cat(x)
        
        return x

class GCN_Low(nn.Module):
    def __init__(self, features_size, embedding_size, bias=False):
        super(GCN_Low, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):
        output = torch.spmm(adj, feature)
        output = 0.5 * output + 0.5 * feature
        output = torch.mm(output, self.weight)
        
        if self.bias is not None:
            output += self.bias
        return output

class GCN_Mid(nn.Module):
    def __init__(self, features_size, embedding_size, bias=False):
        super(GCN_Mid, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):
        output = torch.spmm(adj, feature)
        output = torch.spmm(adj, output)
        output = 0.5 * output - 0.5 * feature
        output = torch.mm(output, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

class Item_Graph_Convolution(nn.Module):
    def __init__(self, features_size, embedding_size, mode):
        super(Item_Graph_Convolution, self).__init__()
        self.mode = mode
        self.gcn_low = GCN_Low(features_size, embedding_size)
        self.gcn_mid = GCN_Mid(features_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        if mode == "concat":
            self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        else:
            self.nn_cat = None

    def forward(self, feature, adj):
        output_low = self.bn1(self.gcn_low(feature, adj))
        output_mid = self.bn2(self.gcn_mid(feature, adj))
        
        if self.mode == "att":
            output = torch.cat([torch.unsqueeze(output_low, dim=1), 
                                torch.unsqueeze(output_mid, dim=1)], dim=1)
        elif self.mode == "concat":
            output = self.nn_cat(torch.cat([output_low, output_mid], dim=1))
        elif self.mode == "mid":
            output = output_mid
        else:
            output = output_low
            
        return output

class SComGNNModel(GeneralModel):
    @staticmethod
    def parse_model_args(parser):
        # 不在这里添加参数，所有参数都在主脚本中定义
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        
        # Store model hyperparameters
        self.embedding_dim = args.embedding_dim
        self.price_n_bins = args.price_n_bins
        self.category_emb_size = args.category_emb_size
        self.mode = args.mode
        
        # 显式存储 corpus，因为父类可能没有存储
        self.corpus = corpus
        
        # Get graph data from corpus
        graph_data = corpus.get_graph_data()
        self.features = torch.FloatTensor(graph_data['features']).to(self.device)
        self.price_bin = torch.LongTensor(graph_data['price_bin']).to(self.device)
        self.adj = graph_data['adj'].to(self.device)
        self.category_emb = torch.FloatTensor(graph_data['category_emb']).to(self.device)
        
        # Initialize model components
        self._define_params()
        
    def _define_params(self):
        """Initialize all model parameters"""
        self.embedding_cid2 = nn.Linear(self.category_emb_size, self.embedding_dim, bias=True)
        self.embedding_cid3 = nn.Linear(self.category_emb_size, self.embedding_dim, bias=True)
        self.embedding_price = nn.Embedding(self.price_n_bins, self.embedding_dim)
        self.nn_emb = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        self.item_gc = Item_Graph_Convolution(self.embedding_dim, self.embedding_dim, self.mode)
        
        # 添加用户嵌入层
        if hasattr(self, 'corpus'):
            self.user_embedding = nn.Embedding(self.corpus.n_users, self.embedding_dim)
        else:
            # 如果corpus不可用，创建一个占位符（在实际训练时会失败）
            self.user_embedding = nn.Embedding(10000, self.embedding_dim)
        
        if self.mode == "att":
            self.two_att = Twostage_Attention(self.embedding_dim)
        
        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, feed_dict: Dict) -> Dict:
        """
        Forward pass for SComGNN
        """
        user_ids = feed_dict['user_id']  # [batch_size]
        item_ids = feed_dict['item_id']  # [batch_size, 1+num_neg]
        batch_size = len(user_ids)
        
        # Get item embeddings
        item_embeddings = self._get_item_embeddings()
        
        # 检查 item_embeddings 的维度并调整
        if len(item_embeddings.shape) == 3:
            # 如果是3维，可能是 [num_items, 2, embedding_dim] 对于 att 模式
            # 取第二个维度的平均值（第1维是batch，第2维是attention heads）
            item_embeddings_2d = item_embeddings.mean(dim=1)
        else:
            item_embeddings_2d = item_embeddings
        
        # Prepare positive and negative item embeddings
        pos_item_idx = item_ids[:, 0]  # First item is positive
        neg_item_idx = item_ids[:, 1:]  # Rest are negatives
        
        pos_emb = item_embeddings_2d[pos_item_idx]  # [batch_size, embedding_dim]
        neg_emb = item_embeddings_2d[neg_item_idx]  # [batch_size, num_neg, embedding_dim]
        
        # 使用用户嵌入层
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        
        # Calculate scores
        if self.mode == "att":
            # 注意：two_att 期望 3D 输入 [batch_size, seq_len, hidden_size]
            # 对于正样本
            pos_emb_expanded = pos_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            user_emb_expanded = user_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            
            key_latent_pos = self.two_att(pos_emb_expanded, user_emb_expanded)
            pos_latent = self.two_att(user_emb_expanded, pos_emb_expanded)
            pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)  # [batch_size]
            
            # 对于负样本
            neg_scores_list = []
            for i in range(self.num_neg):
                neg_emb_i = neg_emb[:, i, :].unsqueeze(1)  # [batch_size, 1, embedding_dim]
                key_latent_neg = self.two_att(neg_emb_i, user_emb_expanded)
                neg_latent = self.two_att(user_emb_expanded, neg_emb_i)
                neg_score_i = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=1)
                neg_scores_list.append(neg_score_i)
            neg_scores = torch.stack(neg_scores_list, dim=1)  # [batch_size, num_neg]
        else:
            # Simple dot product
            pos_scores = torch.sum(torch.mul(user_emb, pos_emb), dim=1)  # [batch_size]
            user_emb_expanded = user_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            neg_scores = torch.sum(torch.mul(user_emb_expanded, neg_emb), dim=2)  # [batch_size, num_neg]
        
        # Concatenate positive and negative scores
        predictions = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        
        return {'prediction': predictions, 'user_id': user_ids, 'item_id': item_ids}

    def _get_item_embeddings(self):
        """Get item embeddings through GCN layers"""
        # Split category embeddings
        cid2 = self.category_emb[:, :self.category_emb_size]
        cid3 = self.category_emb[:, self.category_emb_size:]
        
        # Embed category and price
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(self.price_bin)
        
        # Combine embeddings
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))
        
        # Apply graph convolution
        item_latent = self.item_gc(item_latent, self.adj)
        
        return item_latent

    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index):
            """Get feed dict for training/evaluation"""
            user_id = self.data['user_id'][index]
            target_item = self.data['item_id'][index]
            
            if self.phase == 'train':
                # Get negative items from dataset
                if 'neg_items' in self.data and index < len(self.data['neg_items']):
                    neg_items = self.data['neg_items'][index]
                else:
                    # Sample negative items
                    clicked_set = self.corpus.train_clicked_set.get(user_id, set())
                    neg_items = []
                    while len(neg_items) < self.model.num_neg:
                        neg_item = np.random.randint(1, self.corpus.n_items)
                        if neg_item not in clicked_set:
                            neg_items.append(neg_item)
                    self.data['neg_items'][index] = neg_items
                    
            else:  # dev or test
                # For evaluation, use provided negative items or all items
                if 'neg_items' in self.data and index < len(self.data['neg_items']):
                    neg_items = self.data['neg_items'][index]
                elif self.model.test_all:
                    neg_items = np.arange(1, self.corpus.n_items)
                else:
                    # Sample fixed number of negative items
                    clicked_set = self.corpus.train_clicked_set.get(user_id, set()) | \
                                 self.corpus.residual_clicked_set.get(user_id, set())
                    neg_items = []
                    while len(neg_items) < self.model.num_neg:
                        neg_item = np.random.randint(1, self.corpus.n_items)
                        if neg_item not in clicked_set:
                            neg_items.append(neg_item)
            
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }
            
            return feed_dict

        def actions_before_epoch(self):
            """Sample negative items before each training epoch"""
            if self.phase == 'train':
                # Initialize or refresh negative items
                if 'neg_items' not in self.data:
                    self.data['neg_items'] = [[] for _ in range(len(self))]
                
                for i, u in enumerate(self.data['user_id']):
                    clicked_set = self.corpus.train_clicked_set.get(u, set())
                    neg_items = []
                    
                    # Sample unique negative items
                    while len(neg_items) < self.model.num_neg:
                        neg_item = np.random.randint(1, self.corpus.n_items)
                        if neg_item not in clicked_set and neg_item not in neg_items:
                            neg_items.append(neg_item)
                    
                    self.data['neg_items'][i] = neg_items