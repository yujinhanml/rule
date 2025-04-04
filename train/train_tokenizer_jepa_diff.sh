#!/bin/bash
#SBATCH --job-name=jepa_diff_train
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH --output=output_%j.log

# 你环境的设置
source ~/.bashrc
conda activate yjenv  # <<< 替换成你自己的环境名

# 进入你的代码目录
cd /cpfs04/user/hanyujin/rule-gen/

# 创建 log 目录（如果没创建）
mkdir -p logs

# 运行训练
torchrun --nproc_per_node=4 -m train.train_tokenizer_jepa_diff \
  --config  /project/flame/haoc3/rule_tokenizer/rule/configs/in1k/exp006-aejepadiff-16.yaml

