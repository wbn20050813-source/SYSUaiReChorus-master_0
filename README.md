# SComGNN in ReChorus Framework

本项目在[ReChorus推荐系统框架](https://github.com/THUwangcy/ReChorus)下实现了SComGNN（Social Commerce Graph Neural Network）算法。SComGNN是一种结合图神经网络和注意力机制的社交电商推荐算法。

##该readme文件与文件夹内的第一个readme文件相同


##简单运行
用的是SComGNN预处理好的数据集
基本训练命令：
```bash
python run_rechorus.py --dataset meta_Grocery_and_Gourmet_Food --mode concat
```
结构目录在下面 基本所有文件都在src目录下 就在src目录下运行就行







## 🎯 项目概述

SComGNN（Social Commerce Graph Neural Network）是一种用于社交电商推荐的图神经网络模型。该模型通过以下方式提升推荐性能：

1. **多级图卷积**：结合低层和高层图卷积捕捉不同层次的物品关系
2. **双重注意力机制**：使用两阶段注意力机制学习用户-物品交互
3. **多模态特征融合**：整合价格、类别等多维度特征
4. **社交电商特性**：专门针对社交电商场景设计

本实现将SComGNN算法适配到ReChorus框架，使其能够利用ReChorus的标准数据格式、训练流程和评估体系。

## ✨ 主要特性

- ✅ **完整SComGNN实现**：支持所有原始模型变体（concat、mid、low）
- ✅ **ReChorus兼容**：完全集成到ReChorus框架生态
- ✅ **数据格式**：支持SComGNN原生NPZ格式
- ✅ **标准化评估**：使用ReChorus标准评估指标（HR、NDCG等）
- ✅ **易于扩展**：模块化设计，便于添加新功能
- ✅ **详细日志**：完整的训练日志和实验记录
- ✅ **模型保存/加载**：支持断点续训和模型重用

## 📁 项目结构

```
src/
├── helpers/                    # ReChorus助手模块
│   ├── BaseReader.py          # 基础数据读取器
│   ├── BaseRunner.py          # 基础训练运行器
│   ├── SComGNNReader.py       # SComGNN数据读取器
│   └── SComGNNRunner.py       # SComGNN训练运行器
├── models/                    # 模型定义
│   ├── BaseModel.py          # 基础模型类
│   ├── SComGNNModel.py       # SComGNN模型实现
│   └── __init__.py
├── utils/                     # 工具函数
│   └── utils.py              # 通用工具函数
├── data_preprocess/          # 数据预处理目录
│   ├── processed/           # 处理后的数据
│   │   └── *.npz           # SComGNN格式数据集
│   └── embs/                # 预训练嵌入
│       └── *_embeddings.npz # 类别嵌入文件
├── logs/                     # 日志文件目录
├── model/                    # 模型保存目录
├── config_scomgnn.yaml      # 配置文件示例
├── run_rechorus.py         # 主运行脚本
```

## 🚀 快速开始

### 环境要求

ReChorus的requirements.txt一键配置

### 数据准备

 **使用SComGNN格式数据**：
   ```bash
   # 数据应放在以下目录结构
   data_preprocess/
   ├── processed/
   │   └── {dataset_name}.npz      # 包含features, com_edge_index, train_set, val_set, test_set
   └── embs/
       └── {dataset_name}_embeddings.npz  # 包含cid2_emb, cid3_emb
   ```


## 📖 使用指南

### 训练模型

基本训练命令：
```bash
python run_rechorus.py --dataset meta_Grocery_and_Gourmet_Food --mode concat
```

完整参数训练：
```bash
python run_rechorus.py \
  --dataset meta_Grocery_and_Gourmet_Food \
  --mode concat \
  --embedding_dim 32 \
  --lr 0.001 \
  --batch_size 512 \
  --epoch 300 \
  --num_neg 20 \
  --log_file logs/scomgnn_experiment.log
```

### 参数配置

#### 通过配置文件运行
```bash
python run_rechorus.py @config_scomgnn.yaml
```

#### 主要参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|---------|------|
| `--dataset` | str | `"Appliances"` | 数据集名称 |
| `--mode` | str | `"concat"` | 模型变体: concat/concat/mid/low |
| `--embedding_dim` | int | `16` | 嵌入维度 |
| `--lr` | float | `0.005` | 学习率 |
| `--batch_size` | int | `256` | 批大小 |
| `--epoch` | int | `200` | 训练轮数 |
| `--num_neg` | int | `10` | 负样本数 |
| `--device` | str | `"cuda:0"` | 运行设备 |

## 🔧 算法细节

### 模型架构

SComGNN模型包含以下核心组件：

1. **特征提取层**：
   - 类别特征嵌入（cid2, cid3）
   - 价格特征分箱与嵌入
   - 特征拼接与非线性变换

2. **图卷积层**：
   - `GCN_Low`：低层图卷积，捕捉局部结构
   - `GCN_Mid`：中层图卷积，捕捉全局结构
   - `Item_Graph_Convolution`：组合不同层级的GCN输出

3. **注意力机制**（仅concat模式）：
   - `Twostage_concatention`：两阶段注意力机制
   - 第一阶段：物品间配对注意力
   - 第二阶段：自注意力

4. **预测层**：
   - 用户-物品交互得分计算
   - BPR损失函数优化

### 超参数设置

#### 训练超参数
```python
# 推荐的超参数配置
hyperparams = {
    'learning_rate': 0.001,      # 学习率
    'weight_decay': 5e-8,        # L2正则化
    'batch_size': 256,           # 批大小
    'epochs': 200,               # 训练轮数
    'embedding_dim': 32,         # 嵌入维度
    'num_negatives': 10,         # 负样本数
    'dropout': 0.2,              # Dropout率
}
```

#### 模型架构超参数
```python
# 不同模式的超参数建议
mode_configs = {
    
    'concat': {   # 拼接模式
        'description': '拼接不同GCN层的输出',
        'complexity': '中',
        'best_for': '多特征融合场景'
    },
    'mid': {      # 中层GCN模式
        'description': '仅使用中层GCN',
        'complexity': '低',
        'best_for': '全局关系重要场景'
    },
    'low': {      # 低层GCN模式
        'description': '仅使用低层GCN',
        'complexity': '低',
        'best_for': '局部关系重要场景'
    }
}
```
