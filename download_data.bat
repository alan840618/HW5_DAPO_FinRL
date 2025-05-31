@echo off
REM 創建資料夾
if not exist ".\dataset" (
    mkdir .\dataset
)
if not exist ".\checkpoint" (
    mkdir .\checkpoint
)

REM 下載數據集
echo Downloading datasets...
REM 測試集
curl -L -o .\dataset\trade_data_deepseek_risk_2019_2023.csv https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/resolve/main/trade_data_deepseek_risk_2019_2023.csv
curl -L -o .\dataset\trade_data_deepseek_sentiment_2019_2023.csv https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/resolve/main/trade_data_deepseek_sentiment_2019_2023.csv
REM 訓練集
curl -L -o .\dataset\trade_data_deepseek_risk_2013_2018.csv https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/resolve/main/train_data_deepseek_risk_2013_2018.csv
curl -L -o .\dataset\trade_data_deepseek_sentiment_2013_2018.csv https://huggingface.co/datasets/benstaf/nasdaq_2013_2023/resolve/main/train_data_deepseek_sentiment_2013_2018.csv

REM 下載模型檢查點
echo Downloading model checkpoint...
curl -L -o .\checkpoint\model_rl.pth https://huggingface.co/rz2689/finrl-dapo-grpo-sentiment-risk/resolve/main/model_rl.pth

echo Download completed!
pause
