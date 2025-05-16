# 基于对比学习的RAG检索增强生成系统

本项目实现了一个结合对比学习（Contrastive Learning）与检索增强生成（Retrieval-Augmented Generation, RAG）的问答系统，旨在提升文档检索与生成答案的准确性。系统支持中英文数据，具备数据采集、对比学习训练、向量检索、RAG生成与评测等完整流程。

## 目录结构

```
.
├── contrastive/                # 对比学习相关代码
│   ├── main.py                 # 对比学习主流程入口
│   ├── model.py                # 对比学习模型实现
│   ├── train.py                # 对比学习训练脚本
│   ├── retrieval.py            # 对比学习检索脚本
│   ├── README.md               # 对比学习子模块说明
│   └── ...                     
├── data/                       # 存放中英文原始与处理后数据集
├── model/                      # 存放预训练模型及相关文件
├── results/                    # 存放实验结果
├── RAG.py                      # RAG主流程与评测脚本
├── vector_retrieval.py         # 向量检索与BM25检索实现
├── models.py                   # 支持多种大语言模型的统一接口
├── utils.py                    # 数据处理与工具函数
├── get_data.py                 # 新闻网页爬取与问答生成
├── get_data_bbc.py             # BBC新闻数据采集脚本
├── google_search.py            # 谷歌搜索API相关脚本
├── run_contrastive.sh          # 一键运行脚本
└── instruction.yaml            # 评测用Prompt模板
```

## 主要功能

- **数据采集与处理**：支持从新闻网站、谷歌搜索等多渠道采集中英文问答数据，并自动生成正负样本。
- **对比学习训练**：通过对比学习方法训练检索模型，将query和文档编码到同一语义空间，提升相关性判别能力。
- **向量检索与BM25检索**：支持基于SentenceTransformer的向量检索与BM25文本检索，适配中英文。
- **RAG生成与评测**：集成多种大语言模型（如Qwen、ChatGLM、Baichuan等），支持多种Prompt模板，自动评测生成结果的准确率、F1等指标。
- **全流程自动化**：一键运行训练、检索、评测全流程，支持灵活参数配置。

## 安装依赖

请确保已安装以下主要依赖（部分依赖需根据实际模型和环境补充）：

- Python 3.8+
- torch
- sentence-transformers
- faiss
- transformers
- tqdm
- spacy
- jieba
- nltk
- requests
- beautifulsoup4
- pyyaml

安装示例：

```bash
pip install torch sentence-transformers faiss-cpu transformers tqdm spacy jieba nltk requests beautifulsoup4 pyyaml
python -m spacy download zh_core_web_sm
```

## 快速开始

### 1. 数据采集与处理

- 运行 `get_data.py` 或 `get_data_bbc.py` 可自动爬取新闻网页并生成问答数据，数据保存在 `data/` 目录下。
- 可用 `utils.py` 进行正负样本划分、格式转换等数据预处理。

### 2. 对比学习训练与检索

- 训练对比学习模型：
  ```bash
  python contrastive/main.py --train --data_path data/zh.json
  ```
- 用训练好的模型进行检索：
  ```bash
  python contrastive/main.py --retrieve --data_path data/zh.json
  ```

### 3. RAG生成与评测

- 评测检索增强生成效果：
  ```bash
  python contrastive/main.py --evaluate --dataset zh --modelname Qwen2.5 --method TA_ARE
  ```
- 或直接运行全流程（训练+检索+评测）：
  ```bash
  python contrastive/main.py --data_path data/zh.json --dataset zh
  ```

### 4. 其他脚本

- `vector_retrieval.py`：独立的向量检索与BM25检索实现，可用于检索算法对比实验。
- `RAG.py`：支持多种大模型的RAG主流程与评测脚本，参数详见脚本内说明。

## 参数说明

主要参数如下，更多参数及默认值详见 `contrastive/README.md` 或各脚本内帮助信息：

- `--train` / `--retrieve` / `--evaluate`：控制流程阶段
- `--data_path`：数据文件路径
- `--dataset`：评测用数据集名
- `--base_model`：句向量基模型路径
- `--modelname`：评测用大模型名称
- `--method`：评测方法（Prompt模板）
- `--output_dir` / `--model_dir`：输出与模型保存目录

## 结果与评测

- 检索与生成结果保存在 `results/` 目录下，包含准确率、F1、召回率等多项指标。
- 支持对比不同检索方法、不同大模型、不同Prompt策略的效果。

## 贡献与扩展

- 支持自定义数据集、扩展新模型、替换检索算法等，欢迎贡献代码与建议！

如需详细了解对比学习模块，请参考 `contrastive/README.md`。如有问题欢迎提issue或联系作者。 