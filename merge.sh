# 创建目标目录（如果不存在）
mkdir -p /cpfs04/user/hanyujin/rule-gen/datasets/cifar-mnist-sample-merge

# 进入源目录
cd /cpfs04/user/hanyujin/rule-gen/datasets/cifar-mnist-sample

# 使用循环遍历所有子文件夹并复制文件
for subdir in */; do
    # 获取子文件夹名称（去除末尾斜杠）
    dirname=$(basename "$subdir")
    
    # 遍历子文件夹内的所有文件
    find "$subdir" -type f -print0 | while IFS= read -r -d $'\0' file; do
        # 获取原文件名
        filename=$(basename "$file")
        
        # 生成新文件名（子文件夹_原文件名）
        newname="${dirname}_${filename}"
        
        # 复制文件到目标目录
        cp -v "$file" "/cpfs04/user/hanyujin/rule-gen/datasets/cifar-mnist-sample-merge/$newname"
    done
done