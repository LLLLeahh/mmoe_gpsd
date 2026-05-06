# kuairec_mmoe

使用KuaiRec 公开数据集，基于GPSD实现MMOE多目标预估
**因果 Transformer 序列预训练** + **MMoE 多任务判别式排序**（CTR + `watch_ratio`），支持从预训练 checkpoint **按层名选择性加载并冻结 `item_embeddings`**（ST&SF 思路）。

实现栈：**TensorFlow 2.13 / Keras**，**gin-config** 管理超参。

## 功能概览

| 阶段 | 说明 |
|------|------|
| 数据预处理 | 大/小矩阵、因果划分、流行度负采样、`feature_config`、Parquet 落盘 |
| 序列构建 | 正样本点击序列 → `pretrain_sequences.npz` |
| 生成式预训练 | 因果 Transformer，next-item / next-cate，weight-tied item logits |
| 判别式训练 | MMoE 双塔：CTR（sigmoid）+ watch_ratio（线性回归 + MAE） |
| 迁移实验 | `load_params` / `frozen_params` 正则匹配层名，典型：只迁/冻 `item_embeddings` |

## 数据目录

将 KuaiRec 原始表放入 `data/raw/`：

| 文件 | 说明 |
|------|------|
| `big_matrix.csv` | 训练 / 验证：脚本内按**用户时间**切分（前 80% / 后 20%），再分别做负采样 |
| `small_matrix.csv` | 测试集（不参与大矩阵内的 train/val 切分） |
| `item_feat.csv` | 视频侧 tag：`feat` 字段解析为多值 ID，固定序列长度后并入样本 |

预处理输出在 `data/processed/`，主要包括：

- `train.parquet` / `val.parquet` / `test.parquet`
- `feature_config.pkl`（稀疏/稠密/tag 列与词表，供 MMoE 构图）
- `encoders.pkl`、`scaler.pkl`
- `item_tags.csv`、`tag_vocab.pkl`

## 快速开始

### 1. 预处理

```bash
python data/data_preprocessing.py

默认 data_path='data/raw/'，output_path='data/processed/'；如需修改，编辑 data/data_preprocessing.py 末尾 KuaiRecPreprocessor(...) 参数。

### 2. 构建预训练序列
```bash
python data/build_kuairec_sequences.py \
  --train_csv data/processed/train.parquet \
  --out data/processed/pretrain_sequences.npz \
  --max_seq_len 200
```
（Windows PowerShell 若不支持行尾 `\`，可改为单行执行。）
### 3. Transformer 序列预训练
```bash
python -m src.train --config config/kuairec/transformer_pretrain.gin
```
权重默认落在 `output/kuairec/transformer_pretrain/<时间戳>/`，其中包含 **`weights_final.weights.h5`**。
### 4. MMoE 判别式 baseline（不加载预训练）
```bash
python -m src.train --config config/kuairec/mmoe_baseline.gin
```
### 5. 加载预训练 `item_embeddings` 并冻结（ST&SF 风格）
将 `config/kuairec/mmoe_stsf.gin` 中的 `train.load_ckpt` 指向上一步生成的 `weights_final.weights.h5`，然后执行：
```bash
python -m src.train --config config/kuairec/mmoe_stsf.gin
```
该配置示例中：`load_params` 与 `frozen_params` 均为 `.*item_embeddings.*`（**只加载并冻结物品大表嵌入**，MMoE / 其它稀疏表 / 塔等继续训练）。
### 可选：用户序列长度 EDA
```bash
python data/analyze_user_seq_length.py --matrix big --out-dir output
```
在 `output/` 下生成分位数等统计（及可选直方图），用于评估序列截断长度等。
## 配置说明（入口）
| 文件 | 用途 |
|------|------|
| `config/kuairec/transformer_pretrain.gin` | 序列预训练；通常 `rank_loss_weight=0` |
| `config/kuairec/mmoe_baseline.gin` | 纯 MMoE，不依赖预训练 ckpt |
| `config/kuairec/mmoe_stsf.gin` | 加载预训练 `item_embeddings` 并冻结 |
| `config/kuairec/mmoe_st.gin` 等 | 其它实验配方（按需） |
通用训练项（学习率、Warmup/Cosine、梯度裁剪、混合精度等）见各 `.gin` 与 `src/train.py`。
## 参考
- **KuaiRec** 数据集。
- **GPSD**：*Scaling Transformers for Discriminative Recommendation via Generative Pretraining*；官方 PyTorch 实现见 [gpsd-rec](https://github.com/chqiwang/gpsd-rec)。本仓库为 **KuaiRec + TensorFlow** 的复现与扩展，细节以本仓库代码为准。
