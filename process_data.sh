#!/bin/bash
start_batch=100
end_batch=134
dataset_url="https://huggingface.co/datasets/cs-mshah/SynMirror/resolve/main"
extract_dir="extracted_batch_1"  # 固定解压目录名称
log_dir="download_logs"

mkdir -p $log_dir

for ((i=start_batch; i<=end_batch; i++)); do
    tar_file="batch_${i}.tar"
    log_file="${log_dir}/batch_${i}.log"

    echo "▶▶ 处理批次 $i ◀◀"
    
    # 下载环节
    echo "下载: $tar_file"
    if ! curl -#L -o "$tar_file" "${dataset_url}/${tar_file}" 2>"$log_file"; then
        echo "❌ 下载失败，检查日志 $log_file"
        continue
    fi

    # 清理残留目录
    [ -d "$extract_dir" ] && rm -rf "$extract_dir"
    
    # 解压环节
    echo "解压到: $extract_dir"
    if ! mkdir -p "$extract_dir" || ! tar -xf "$tar_file" -C "$extract_dir"; then
        echo "❌ 解压失败，清理残留文件"
        rm -f "$tar_file"
        rm -rf "$extract_dir"
        continue
    fi
    
    # 数据处理环节
    echo "启动数据处理..."
    if ! python process_data.py --input_dir "$extract_dir"; then
        echo "❌ 数据处理失败，跳过本批次"
        rm -f "$tar_file"
        rm -rf "$extract_dir"
        continue
    fi
    
    # 清理环节
    echo "清理临时文件..."
    rm -f "$tar_file"
    rm -rf "$extract_dir"
done