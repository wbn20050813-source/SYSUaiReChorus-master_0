# -*- coding: UTF-8 -*-
import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
from sklearn.metrics import ndcg_score

from utils import utils
from helpers.BaseRunner import BaseRunner
from models.BaseModel import BaseModel  # 导入 BaseModel 以使用 Dataset 类

class SComGNNRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        # 不在这里添加参数，所有参数都在主脚本中定义
        return parser

    def __init__(self, args):
        super().__init__(args)
        
    def train(self, data_dict: Dict[str, BaseModel.Dataset]):  # 使用 BaseModel.Dataset 类型
        """
        Training process for SComGNN
        """
        model = data_dict['train'].model
        main_metric_results, dev_results = [], []
        self._check_time(start=True)
        
        best_metric = -np.inf
        patience_counter = 0
        
        try:
            for epoch in range(self.epoch):
                # Training phase
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                
                if np.isnan(loss):
                    logging.info(f"Loss is NaN. Stop training at epoch {epoch + 1}")
                    break
                
                training_time = self._check_time()
                
                # Validation phase
                dev_result = self.evaluate(data_dict['dev'], [5, 10], ['HR', 'NDCG'])
                dev_results.append(dev_result)
                
                # Use NDCG@10 as main metric
                current_metric = dev_result.get('NDCG@10', dev_result.get('HR@10', 0))
                main_metric_results.append(current_metric)
                
                # Log results
                logging_str = f'Epoch {epoch + 1:<5} loss={loss:<.4f} [{training_time:<3.1f} s]'
                logging_str += f' dev=({utils.format_metric(dev_result)})'
                
                # Test phase (if enabled)
                if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                    test_result = self.evaluate(data_dict['test'], [5, 10], ['HR', 'NDCG'])
                    logging_str += f' test=({utils.format_metric(test_result)})'
                
                testing_time = self._check_time()
                logging_str += f' [{testing_time:<.1f} s]'
                
                # Check if current model is the best
                if current_metric > best_metric:
                    best_metric = current_metric
                    model.save_model()
                    logging_str += ' *'
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                logging.info(logging_str)
                
                # Early stopping check
                if patience_counter >= self.early_stop:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n): ")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)
        
        # Load the best model
        model.load_model()
        
        # Find the best epoch
        if main_metric_results:
            best_epoch = main_metric_results.index(max(main_metric_results))
            logging.info(f"\nBest Iter(dev)={best_epoch + 1:<5}\t dev=({utils.format_metric(dev_results[best_epoch])})")
        
        logging.info("Training completed.")

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        """
        Custom evaluation method for SComGNN
        """
        evaluations = {}
        
        # Calculate rank of positive items
        gt_rank = (predictions >= predictions[:, 0:1]).sum(axis=1)
        
        for k in topk:
            hit = (gt_rank <= k)
            
            for metric in metrics:
                key = f'{metric}@{k}'
                
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    # Calculate NDCG
                    relevance = np.zeros_like(predictions)
                    relevance[:, 0] = 1  # Positive item has relevance 1
                    
                    # Sort predictions
                    sorted_idx = np.argsort(-predictions, axis=1)
                    sorted_relevance = np.take_along_axis(relevance, sorted_idx, axis=1)
                    
                    # Calculate DCG
                    ranks = np.arange(1, predictions.shape[1] + 1)
                    discounts = np.log2(ranks + 1)
                    dcg = np.sum(sorted_relevance[:, :k] / discounts[:k], axis=1)
                    
                    # Calculate IDCG (ideal: positive item at first position)
                    idcg = 1.0 / np.log2(2)  # log2(1+1) = log2(2) = 1
                    ndcg = dcg / idcg
                    
                    evaluations[key] = ndcg.mean()
                else:
                    raise ValueError(f'Undefined evaluation metric: {metric}')
        
        return evaluations

    # 在SComGNNRunner.py的print_res方法中添加文件输出
    def print_res(self, dataset: BaseModel.Dataset) -> str:
        """Print final results and save to txt file"""
        result_dict = self.evaluate(dataset, [5, 10, 20], ['HR', 'NDCG'])
        res_str = '(' + utils.format_metric(result_dict) + ')'
        
        # 打印结果到控制台
        logging.info("\n" + "="*60)
        logging.info("Final Test Results:")
        logging.info(f"HR@5:  {result_dict.get('HR@5', 0):.4f}")
        logging.info(f"HR@10: {result_dict.get('HR@10', 0):.4f}")
        logging.info(f"HR@20: {result_dict.get('HR@20', 0):.4f}")
        logging.info(f"NDCG@5:  {result_dict.get('NDCG@5', 0):.4f}")
        logging.info(f"NDCG@10: {result_dict.get('NDCG@10', 0):.4f}")
        logging.info(f"NDCG@20: {result_dict.get('NDCG@20', 0):.4f}")
        logging.info("="*60)
        
        # 输出结果到txt文件
        output_path = f"results_{self.args.dataset}_{self.args.mode}.txt"
        with open(output_path, 'w') as f:
            f.write("Final Test Results:\n")
            f.write(f"HR@5:  {result_dict.get('HR@5', 0):.4f}\n")
            f.write(f"HR@10: {result_dict.get('HR@10', 0):.4f}\n")
            f.write(f"HR@20: {result_dict.get('HR@20', 0):.4f}\n")
            f.write(f"NDCG@5:  {result_dict.get('NDCG@5', 0):.4f}\n")
            f.write(f"NDCG@10: {result_dict.get('NDCG@10', 0):.4f}\n")
            f.write(f"NDCG@20: {result_dict.get('NDCG@20', 0):.4f}\n")
        
        logging.info(f"Results saved to {output_path}")
        return res_str