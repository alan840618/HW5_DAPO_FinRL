# FinRL_Contest_2025

![dapo_results](https://github.com/user-attachments/assets/5dc3d27f-44b1-4fdc-9fc0-9ce95717ed18)

This repo is an implementation built on top of the [FinRL-DeepSeek paper](https://arxiv.org/abs/2502.07393),  
which combines reinforcement learning (RL) and large language models (LLMs) for automated stock trading using price and news data.

It is based on the original [FinRL-DeepSeek codebase](https://github.com/benstaf/FinRL_DeepSeek).  
For more context, check out the [FinRL-Contest documentation](https://finrl-contest.readthedocs.io/en/latest/).

## Installation of dependencies 
run `installation_script.sh` on Ubuntu server (128 GB RAM CPU instance recommended)

## 📊 Datasets and Preprocessing

This project uses stock trading data and financial news for training RL agents with LLM signals.

### 🔧 Dataset Preparation from Scratch

The base dataset is **FNSPID**:  
- [FNSPID on Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID) (see `Stock_news/nasdaq_exteral_data.csv`)  
- [FNSPID GitHub Repo](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)  
- [FNSPID Paper (arXiv)](https://arxiv.org/abs/2402.06698)

To add LLM-generated signals, run:
- `sentiment_deepseek_deepinfra.py`
- `risk_deepseek_deepinfra.py`

These scripts generate:
- [Sentiment Dataset](https://huggingface.co/datasets/benstaf/nasdaq_news_sentiment)
- [Risk Dataset](https://huggingface.co/datasets/benstaf/risk_nasdaq)

Next, process the combined data using:
- `train_trade_data_deepseek_sentiment.py`
- `train_trade_data_deepseek_risk.py`

This produces agent-ready datasets.

---

### 💾 Direct Dataset Download

Skip preprocessing and directly download the full dataset from:  
👉 [benstaf/nasdaq_2013_2023](https://huggingface.co/datasets/benstaf/nasdaq_2013_2023)

Download the following files to the `./dataset` folder:

```
trade_data_2019_2023.csv  
trade_data_deepseek_risk_2019_2023.csv  
trade_data_deepseek_sentiment_2019_2023.csv  
train_data_2013_2018.csv  
train_data_deepseek_risk_2013_2018.csv  
train_data_deepseek_sentiment_2013_2018.csv
```

## 🏋️‍♂️ Training and Environments

To start training, run:

```bash
python train_dapo_llm_risk.py --adjustment_type both --alpha 1.0 --beta 1.0
```

The trained model from this command is available at:  
👉 [model_rl.pth on Hugging Face](https://huggingface.co/rz2689/finrl-dapo-grpo-sentiment-risk/blob/main/model_rl.pth)

---

## ✅ Evaluation

To evaluate the trained agent, run:

```bash
python becktest_main_dapo.py
```
