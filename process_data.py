import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def extract_data_from_hdf5(hdf5_path: str):
    """从 hdf5 文件中提取数据"""
    with h5py.File(hdf5_path, "r") as hdf5_data:
        data = {
            "image": np.array(hdf5_data["colors"], dtype=np.uint8),
            "mask": (np.array(hdf5_data["category_id_segmaps"], dtype=np.uint8) == 1).astype(np.uint8) * 255,
            "object_mask": (np.array(hdf5_data["category_id_segmaps"], dtype=np.uint8) == 2).astype(np.uint8) * 255,
            "depth": np.array(hdf5_data["depth"]),
            "normals": np.array(hdf5_data["normals"]),
            "cam_states": np.array(hdf5_data["cam_states"]),
        }
    return data

def main():
    extract_dir = "extracted_batch_1"
    dest_root = "/cpfs04/user/hanyujin/rule-gen/datasets/mirrors"
    dest_dirs = {
        "middle": os.path.join(dest_root, "middle"),
        "left": os.path.join(dest_root, "left"),
        "right": os.path.join(dest_root, "right")
    }

    # 创建目标目录（如果不存在）
    for folder in dest_dirs.values():
        os.makedirs(folder, exist_ok=True)

    # 初始化计数器：动态查找每个目录中的最大现有编号
    counters = {}
    for category, dir_path in dest_dirs.items():
        max_num = 0
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.lower().endswith('.png'):
                    name_part = os.path.splitext(filename)[0]
                    if name_part.isdigit():  # 仅处理纯数字文件名
                        current_num = int(name_part)
                        max_num = max(max_num, current_num)
        counters[category] = max_num

    # 收集所有 hdf5 文件
    hdf5_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".hdf5"):
                hdf5_files.append(os.path.join(root, file))

    # 处理每个 HDF5 文件
    for hdf5_path in tqdm(hdf5_files, desc="Processing HDF5 files"):
        try:
            data = extract_data_from_hdf5(hdf5_path)
            image = Image.fromarray(data["image"]).resize((128, 128), Image.Resampling.LANCZOS)

            # 根据文件名前缀确定分类
            base_name = os.path.basename(hdf5_path)
            if base_name.startswith("0"):
                category = "middle"
            elif base_name.startswith("1"):
                category = "left"
            elif base_name.startswith("2"):
                category = "right"
            else:
                print(f"未知文件名格式: {base_name}, 跳过")
                continue

            # 更新计数器并保存图片
            counters[category] += 1
            out_filename = f"{counters[category]:05d}.png"
            out_path = os.path.join(dest_dirs[category], out_filename)
            image.save(out_path)
        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")

if __name__ == "__main__":
    main()