## 環境安裝(Windows 適用)
1. 先安裝Anaconda prompt
2. 執行`setup.sh` 

## 預處理資料下載

使用download_data.bat下載預處理數據集(已加入LLM判斷訊號)
- [benstaf/nasdaq_2013_2023](https://huggingface.co/datasets/benstaf/nasdaq_2013_2023)

### 未處理資料下載

原始資料集 **FNSPID**:  
- [FNSPID on Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID) (see `Stock_news/nasdaq_exteral_data.csv`)  
- [FNSPID GitHub Repo](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)  
- [FNSPID Paper (arXiv)](https://arxiv.org/abs/2402.06698)

### 資料處理加入LLM市場訊號

加入LLM產生之市場訊號(情緒、風險):
- `sentiment_deepseek_deepinfra.py`
- `risk_deepseek_deepinfra.py`

訓練及測試資料整合:
- `train_trade_data_deepseek_sentiment.py`
- `train_trade_data_deepseek_risk.py`

完成後將資料放入`./dataset`



### Training and Environments

訓練模型指令

```bash
python train_dapo_llm_risk.py --adjustment_type both --alpha 1.0 --beta 1.0
```

供參考:使用download_data.bat會一併下載已訓練權重至 `./checkpoint`
```
model_rl.pth
```

### Evaluation

模型測試指令:

```bash
python backtest_main_dapo.py
```
以上內容均參考自:https://github.com/Ruijian-Zha/FinRL-DAPO-SR/tree/main