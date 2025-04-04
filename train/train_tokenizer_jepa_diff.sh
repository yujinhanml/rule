#!/bin/bash
#SBATCH --job-name=jepa_diff_train
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/error_%j.err

set -x
set -e

# ✅ 初始化 conda
source /project/flame/haoc3/conda3/etc/profile.d/conda.sh
conda activate yjenv

cd /project/flame/haoc3/rule_tokenizer/rule/
mkdir -p logs

# 启动训练
torchrun --nproc_per_node=4 -m train.train_tokenizer_jepa_diff \
  --config /project/flame/haoc3/rule_tokenizer/rule/configs/in1k/exp006-aejepadiff-16.yaml
