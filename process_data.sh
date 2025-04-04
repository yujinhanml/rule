#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH --output=output_%j.log

# 实时日志刷新
exec > >(stdbuf -oL tee slurm_log.out) 2>&1

start_batch=1
end_batch=134
dataset_url="https://huggingface.co/datasets/cs-mshah/SynMirror/resolve/main"
extract_dir="extracted_batch_1"
log_dir="/project/flame/haoc3/rule_tokenizer/rule/datasets/mirrors_download_logs"
output_dir="/project/flame/haoc3/rule_tokenizer/rule/datasets/mirrors"

mkdir -p $log_dir

for ((i=start_batch; i<=end_batch; i++)); do
    tar_file="batch_${i}.tar"
    log_file="${log_dir}/batch_${i}.log"

    echo "▶▶ 处理批次 $i ◀◀"
    
    echo "下载: $tar_file"
    if ! curl -#L -o "$tar_file" "${dataset_url}/${tar_file}" 2>"$log_file"; then
        echo "❌ 下载失败，检查日志 $log_file"
        continue
    fi

    [ -d "$extract_dir" ] && rm -rf "$extract_dir"
    
    echo "解压到: $extract_dir"
    if ! mkdir -p "$extract_dir" || ! tar -xf "$tar_file" -C "$extract_dir"; then
        echo "❌ 解压失败，清理残留文件"
        rm -f "$tar_file"
        rm -rf "$extract_dir"
        continue
    fi
    
    echo "启动数据处理..."
    if ! python -u process_data.py --input_dir "$extract_dir" --output_dir "$output_dir"; then
        echo "❌ 数据处理失败，跳过本批次"
        rm -f "$tar_file"
        rm -rf "$extract_dir"
        continue
    fi
    
    echo "清理临时文件..."
    rm -f "$tar_file"
    rm -rf "$extract_dir"
done
