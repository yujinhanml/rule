import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
# -------------------- 设备配置 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- 配置文件 --------------------
dataset_root = "/cpfs04/user/hanyujin/rule-gen/datasets/cifar-mnist"
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
mnist_classes = [str(i) for i in range(10)]

# -------------------- CIFAR模型定义 --------------------
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.features = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 256x4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
   
# -------------------- MNIST模型定义 --------------------
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            # 输入: 1x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道改为1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 输出: 32x14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 输出: 64x7x7
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化输出: 128x1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -------------------- 数据集类 --------------------
class CompositeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path) and len(folder.split('-')) == 4:
                parts = folder.split('-')
                if (all(p in cifar_classes for p in parts[:2]) and 
                    all(p in mnist_classes for p in parts[2:])):
                    for img in os.listdir(folder_path):
                        if img.endswith(('.png', '.jpg')):
                            self.samples.append({
                                "path": os.path.join(folder_path, img),
                                "folder": folder,
                                "true_labels": parts
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "path": self.samples[idx]["path"],
            "folder": self.samples[idx]["folder"],
            "true_labels": self.samples[idx]["true_labels"]
        }

# -------------------- 验证器类 --------------------
class FolderValidator:
    def __init__(self, cifar_model, mnist_model):
        self.cifar_model = cifar_model.to(device).eval()
        self.mnist_model = mnist_model.to(device).eval()
        
        self.cifar_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2470, 0.2435, 0.2616))
        ])
        
        self.mnist_tf = transforms.Compose([
            transforms.Resize(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def validate(self, dataloader):
        results = defaultdict(lambda: {"correct":0, "total":0})
        
        for batch in tqdm(dataloader, desc="Validating", unit="img"):
            img_path = batch["path"][0]
            folder = batch["folder"][0]
            true_labels = set(batch["true_labels"][0])
            
            pred_labels = self._predict(img_path)
            results[folder]["total"] += 1
            if pred_labels == true_labels:
                results[folder]["correct"] += 1
        
        return self._compile_report(results)

    def _predict(self, img_path):
        blocks = self._split_image(img_path)
        return self._analyze_blocks(blocks)

    def _split_image(self, path):
        img = transforms.ToTensor()(Image.open(path)).to(device)
        return [
            img[:, :32, :32],   # Top-Left
            img[:, :32, 32:],   # Top-Right
            img[:, 32:, :32],   # Bottom-Left
            img[:, 32:, 32:]    # Bottom-Right
        ]

    def _analyze_blocks(self, blocks):
        diffs = []
        for idx, block in enumerate(blocks):
            with torch.no_grad():
                cifar_conf = self._get_conf(block, "cifar")
                mnist_conf = self._get_conf(block, "mnist")
                diffs.append((idx, mnist_conf - cifar_conf))
        
        mnist_ids = {x[0] for x in sorted(diffs, key=lambda x: -x[1])[:2]}
        
        labels = []
        for idx, block in enumerate(blocks):
            if idx in mnist_ids:
                input_tensor = self.mnist_tf(block).unsqueeze(0)
                pred = str(self.mnist_model(input_tensor).argmax().item())
            else:
                input_tensor = self.cifar_tf(block).unsqueeze(0)
                pred = cifar_classes[self.cifar_model(input_tensor).argmax().item()]
            labels.append(pred)
        return set(labels)

    def _get_conf(self, block, model_type):
        tf = self.cifar_tf if model_type == "cifar" else self.mnist_tf
        model = self.cifar_model if model_type == "cifar" else self.mnist_model
        with torch.no_grad():
            output = torch.nn.functional.softmax(model(tf(block).unsqueeze(0)), dim=1)
        return output.max().item()

    def _compile_report(self, results):
        return {
            folder: {
                "accuracy": stats["correct"] / stats["total"],
                "samples": stats["total"]
            } for folder, stats in results.items()
        }

# -------------------- 报告生成 --------------------
def generate_report(report):
    total_samples = sum(v["samples"] for v in report.values())
    total_correct = sum(int(v["accuracy"]*v["samples"]) for v in report.values())
    
    print("\n{:<30} {:<10} {:<10}".format("Folder", "Accuracy", "Samples"))
    print("-"*50)
    for folder, data in report.items():
        print(f"{folder:<30} {data['accuracy']:.2%}     {data['samples']:<10}")
    
    print("\nOverall Accuracy: {:.2%}".format(total_correct/total_samples))
    print(f"Total Samples: {total_samples}")

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    # 初始化模型
    cifar_model = CIFAR_CNN()
    cifar_model.load_state_dict(torch.load(
        "/cpfs04/user/hanyujin/rule-gen/model_cpkt/cifar_class_cnn_best.pth",
        map_location=device
    ))
    
    mnist_model = MNIST_CNN()
    mnist_model.load_state_dict(torch.load(
        "/cpfs04/user/hanyujin/rule-gen/model_cpkt/mnist_cnn_best.pth",
        map_location=device
    ))

    # 准备数据
    dataset = CompositeDataset(dataset_root)
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        num_workers=8,
        pin_memory=True
    )

    # 执行验证
    validator = FolderValidator(cifar_model, mnist_model)
    report = validator.validate(dataloader)
    
    # 生成报告
    generate_report(report)