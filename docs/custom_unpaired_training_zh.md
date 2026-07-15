# 自定义非配对数据集训练脚本

本文说明如何使用 `scripts/train_custom_unpaired.sh` 训练自己的 source/target
非配对图像数据集。

## 1. 准备环境

在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

项目依赖已匹配 PyTorch 2.0.0 和 torchvision 0.15.1。

## 2. 准备数据

默认目录结构如下：

```text
my_dataset/
├── train/
│   ├── source/
│   │   ├── 001.png
│   │   └── ...
│   └── target/
│       ├── a.png
│       └── ...
└── val/
    ├── source/
    │   └── ...
    └── target/
        └── ...
```

source 和 target 是两个独立域，文件名和图片数量都不需要对应。训练时使用
`train`，验证时直接使用独立的 `val`，不会从训练集再次划分验证集。没有
`test` 目录时，测试加载器会复用 `val`。

支持 PNG、JPG、JPEG、BMP、TIFF 和 WebP，默认递归查找子目录中的图片。

## 3. 启动训练

数据目录位于项目内时：

```bash
./scripts/train_custom_unpaired.sh data/my_dataset
```

数据目录位于项目外时，建议传入绝对路径：

```bash
./scripts/train_custom_unpaired.sh /absolute/path/to/my_dataset
```

脚本参数依次为：

```text
train_custom_unpaired.sh DATA_ROOT SOURCE_DOMAIN TARGET_DOMAIN CONFIG_PATH
```

其中只有 `DATA_ROOT` 通常需要指定，其他参数的默认值为：

```text
SOURCE_DOMAIN=source
TARGET_DOMAIN=target
CONFIG_PATH=configs/custom_unpaired.example.yaml
```

如果域目录使用其他名称，例如 `samsung` 和 `iphone`：

```bash
./scripts/train_custom_unpaired.sh \
  /absolute/path/to/my_dataset \
  samsung \
  iphone
```

如果要使用自己的配置文件，同时保持 `source/target` 目录名：

```bash
./scripts/train_custom_unpaired.sh \
  /absolute/path/to/my_dataset \
  source \
  target \
  configs/my_custom_unpaired.yaml
```

## 4. 调整训练配置

复制示例配置后再修改，可以避免覆盖仓库默认配置：

```bash
cp configs/custom_unpaired.example.yaml configs/my_custom_unpaired.yaml
```

常用参数位于 `configs/my_custom_unpaired.yaml`：

- `data.params.crop_size`：送入模型的裁剪尺寸。
- `data.params.resize_size`：裁剪前的缩放尺寸，不能小于 `crop_size`。
- `data.params.num_workers`：数据加载进程数；遇到多进程问题时可设为 `0`。
- `pipeline.params.batch_size`：训练 batch size，显存不足时调小。
- `pipeline.params.lr`：学习率。
- `pipeline.params.epochs`：训练轮数。
- `accelerator`：使用 `auto` 自动选择设备，也可以设为 `cpu` 或 `gpu`。

训练增强包括缩放、随机裁剪和随机翻转。验证使用缩放与中心裁剪，不使用随机
增强。为了保留两个域的颜色分布，默认不使用颜色抖动。

## 5. 输出与断点续训

默认实验名为 `custom_unpaired`，输出目录为：

```text
experiments/custom_unpaired/
└── logs/
    ├── checkpoints/
    │   └── last.ckpt
    └── metrics.csv
```

需要从 `last.ckpt` 继续训练时，将配置中的 `resume` 改为 `true`，然后再次执行
同一条训练命令。

## 常见问题

- 提示找不到目录：检查是否存在 `train/source`、`train/target`、
  `val/source` 和 `val/target`；项目外的数据建议使用绝对路径。
- 提示没有支持的图片：检查图片扩展名，或在配置的 `image_extensions` 中添加。
- 显存不足：减小 `batch_size`、`val_batch_size` 和裁剪尺寸。
- DataLoader 多进程异常：将 `data.params.num_workers` 设为 `0`。
