REM please run this script in Anaconda Prompt
@echo off
call conda create --name myenv python=3.10 -y
call conda init
call conda activate myenv

call conda install -c conda-forge gcc -y
call conda install -c conda-forge swig -y
call conda install -c conda-forge box2d-py -y
call conda install -c conda-forge mpi4py -y
call conda install -c conda-forge datasets -y
call conda install -c conda-forge huggingface_hub -y
call conda install -c conda-forge alpaca-py -y
call conda install -c conda-forge selenium -y
call conda install -c conda-forge webdriver-manager -y
call pip install gymnasium
call pip install git+https://github.com/benstaf/FinRL.git
git clone https://github.com/benstaf/spinningup_pytorch.git
cd spinningup_pytorch
call pip install -e .
cd ..

echo 安裝完成！請在新的 Conda Prompt 執行 "conda activate myenv" 開始使用。
pause
