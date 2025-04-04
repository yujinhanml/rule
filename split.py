import os
import shutil
import random

def split_dataset(src_dir, train_dir, val_dir, split_ratio=0.8):
    """
    按子文件夹结构分割数据集为训练集和验证集
    :param src_dir: 源目录（包含子文件夹的路径，如 `/project/flame/haoc3/rule_tokenizer/mirrors`）
    :param train_dir: 训练集目标目录（如 `mirrors_train`）
    :param val_dir: 验证集目标目录（如 `mirrors_val`）
    :param split_ratio: 训练集比例（默认 8:2）
    """
    # 遍历每个子文件夹
    for subfolder in os.listdir(src_dir):
        subfolder_path = os.path.join(src_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # 获取子文件夹中的所有文件
            all_files = []
            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
            random.shuffle(all_files)  # 打乱顺序
            
            # 计算分割点
            split_idx = int(len(all_files) * split_ratio)
            train_files = all_files[:split_idx]
            val_files = all_files[split_idx:]
            
            # 创建目标目录结构
            for file in train_files:
                rel_path = os.path.relpath(file, subfolder_path)
                dst_path = os.path.join(train_dir, subfolder, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(file, dst_path)
            
            for file in val_files:
                rel_path = os.path.relpath(file, subfolder_path)
                dst_path = os.path.join(val_dir, subfolder, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(file, dst_path)
            
            # 打印日志
            print(f"子文件夹 [{subfolder}] 分割完成：")
            print(f"  Train: {len(train_files)} 文件")
            print(f"  Val: {len(val_files)} 文件")

if __name__ == "__main__":
    src_dir = "/project/flame/haoc3/rule_tokenizer/mirrors"  # 替换为你的实际路径
    train_dir = "/project/flame/haoc3/rule_tokenizer/mirrors_train"
    val_dir = "/project/flame/haoc3/rule_tokenizer/mirrors_val"
    
    # 清空目标目录（如果已存在）
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    
    # 执行分割
    split_dataset(src_dir, train_dir, val_dir)
