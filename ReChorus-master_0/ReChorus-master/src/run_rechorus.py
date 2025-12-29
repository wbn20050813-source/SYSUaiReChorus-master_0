#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import torch
import logging
import argparse
import numpy as np

# Add current directory to path
sys.path.append('.')
sys.path.append('./helpers')
sys.path.append('./models')

from helpers.SComGNNReader import SComGNNReader
from models.SComGNNModel import SComGNNModel
from helpers.SComGNNRunner import SComGNNRunner
from utils import utils

def parse_args():
    parser = argparse.ArgumentParser(description='SComGNN in ReChorus Framework')
    
    # BaseRunner required arguments
    parser.add_argument('--train', type=int, default=1,
                        help='Whether to train the model.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--check_epoch', type=int, default=1,
                        help='Check some tensors every check_epoch.')
    parser.add_argument('--test_epoch', type=int, default=-1,
                        help='Print test results every test_epoch (-1 means no print).')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='The number of epochs when dev results drop continuously.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--l2', type=float, default=5e-8,
                        help='Weight decay in optimizer.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size during training.')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='Batch size during testing.')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer: SGD, Adam, Adagrad, Adadelta')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of processors when prepare batches in DataLoader')
    parser.add_argument('--pin_memory', type=int, default=0,
                        help='pin_memory in DataLoader')
    parser.add_argument('--topk', type=str, default='5,10,20,50',
                        help='The number of items recommended to each user.')
    parser.add_argument('--metric', type=str, default='NDCG,HR',
                        help='metrics: NDCG, HR')
    parser.add_argument('--main_metric', type=str, default='',
                        help='Main metric to determine the best model.')
    
    # Data arguments
    parser.add_argument('--path', type=str, default='data_preprocess/processed/',
                        help='Input data dir.')
    parser.add_argument('--dataset', type=str, default='Appliances',
                        help='Choose a dataset.')
    parser.add_argument('--sep', type=str, default='\t',
                        help='sep of csv file.')
    parser.add_argument('--price_n_bins', type=int, default=20,
                        help='Number of price bins.')
    parser.add_argument('--category_emb_size', type=int, default=768,
                        help='Category embedding size.')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='Embedding dimension.')
    parser.add_argument('--mode', type=str, default='concat', 
                        choices=['att', 'concat', 'mid', 'low'],
                        help='Model version: att, concat, mid, low')
    parser.add_argument('--num_neg', type=int, default=10,
                        help='Number of negative samples during training.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability.')
    parser.add_argument('--test_all', type=int, default=0,
                        help='Whether testing on all the items.')
    parser.add_argument('--buffer', type=int, default=1,
                        help='Whether to buffer feed dicts for dev/test.')
    parser.add_argument('--model_path', type=str, default='model/',
                        help='Model save path.')
    parser.add_argument('--history_max', type=int, default=20,
                        help='Maximum history length for sequential models.')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use.')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed.')
    parser.add_argument('--log_file', type=str, default='logs/scomgnn.log',
                        help='Log file path.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs('model/', exist_ok=True)
    
    # Set model_path to a specific file if not provided
    if args.model_path == 'model/':
        args.model_path = f'model/SComGNN_{args.dataset}_{args.mode}.pt'
    
    return args

def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    setup_seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Running SComGNN on dataset: {args.dataset}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Model mode: {args.mode}")
    
    try:
        logging.info("Loading data...")
        corpus = SComGNNReader(args)
        args.corpus = corpus  # 传递corpus到args
        
        logging.info("Creating model...")
        model = SComGNNModel(args, corpus)
        model = model.to(args.device)
        
        logging.info("Creating datasets...")
        data_dict = {}
        for phase in ['train', 'dev', 'test']:
            dataset = model.Dataset(model, corpus, phase)
            dataset.prepare()
            data_dict[phase] = dataset
        
        args.data_dict = data_dict  # 传递data_dict到args
        
        logging.info("Creating runner...")
        runner = SComGNNRunner(args)
        
        if args.train:
            logging.info("Starting training...")
            runner.train(data_dict)
        
        # 若不训练，直接评估并保存结果
        else:
            test_result = runner.evaluate(data_dict['test'], 
                                         [int(x) for x in args.topk.split(',')],
                                         [m.strip().upper() for m in args.metric.split(',')])
            logging.info(f"Final test results: {utils.format_metric(test_result)}")
            runner.final_results['test'] = test_result
            runner.save_results_to_txt()
        
        if not os.path.exists(args.model_path):
            model.save_model(args.model_path)
            logging.info(f"Model saved to {args.model_path}")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()