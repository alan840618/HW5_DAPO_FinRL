#!/usr/bin/env python
# coding: utf-8
# run with the command: OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -np 4 python3 train_dapo.py

from datasets import load_dataset
import pandas as pd
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from env_stocktrading_llm_risk import StockTradingEnv
import os
import numpy as np
import torch
import time
import argparse

# Import the DAPO implementation
from dapo_algorithm import run_dapo

# Make necessary directories
check_and_make_directories([TRAINED_MODEL_DIR])

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download and load both risk and sentiment datasets
dataset_dir = "/home/ruijian/FinRL_Contest_2025/Task_1_FinRL_DeepSeek_Stock/dataset"
risk_file = os.path.join(dataset_dir, "train_data_deepseek_risk_2013_2018.csv")
sentiment_file = os.path.join(dataset_dir, "train_data_deepseek_sentiment_2013_2018.csv")

# Download and load risk data
if not os.path.exists(risk_file):
    print(f"Downloading risk dataset to {dataset_dir}...")
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files="train_data_deepseek_risk_2013_2018.csv")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset['train'].to_csv(risk_file, index=False)

# Download and load sentiment data
if not os.path.exists(sentiment_file):
    print(f"Downloading sentiment dataset to {dataset_dir}...")
    dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files="train_data_deepseek_sentiment_2013_2018.csv")
    dataset['train'].to_csv(sentiment_file, index=False)

# Load and merge the datasets
train_risk = pd.read_csv(risk_file)
train_sentiment = pd.read_csv(sentiment_file)

# Merge risk and sentiment data
train = pd.merge(train_risk, train_sentiment, on=['date', 'tic'], suffixes=('', '_sentiment'))

# Create a new index based on unique dates
unique_dates = train['date'].unique()
date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

# Create new index based on the date mapping
train['new_idx'] = train['date'].map(date_to_idx)

# Set this as the index
train = train.set_index('new_idx')

# Fill missing values for both risk and sentiment
train['llm_sentiment'].fillna(3, inplace=True)  # neutral sentiment score is 3
train['llm_risk'].fillna(3, inplace=True)  # neutral risk score is 3

# Set up trading environment parameters
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + (2+len(INDICATORS))*stock_dimension  # add dimensions for LLM sentiment and risk
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

# Create trading environment
e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=env_train)
    parser.add_argument('--hid', type=int, default=512)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='dapo')
    parser.add_argument('-f', '--file', type=str, help='Kernel connection file')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER)
    parser.add_argument('--cpu_only', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID when multiple GPUs are available')
    args = parser.parse_args()
    
    # Important: For single process running without MPI, don't use mpi_fork
    # This will help avoid the CUDA/MPI conflicts
    # If you want to use MPI, run the script with mpirun directly as in the comment at the top
    
    # Train the DAPO agent with our env_kwargs
    trained_dapo = run_dapo(args.env, env_kwargs)
    
    # Save the final model
    checkpoint_dir = "/home/ruijian/FinRL_Contest_2025/Task_1_FinRL_DeepSeek_Stock/checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_model_path = os.path.join(checkpoint_dir, "agent_dapo_deepseek_gpu_final.pth")
    torch.save({
        'epoch': 99,
        'model_state_dict': trained_dapo.state_dict(),
    }, final_model_path)
    print(f"\nTraining finished and final model saved in {final_model_path}")
    print(f"Checkpoints saved in {checkpoint_dir}")