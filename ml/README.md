# Fruit maturity and quality classifiers

This directory contains a PyTorch training script for the two subtasks in the dataset:

- **Maturity** (`Immature`, `Mature`) stored under `Maturity/`
- **Quality** (`Fresh`, `Rotten`) stored under `Quality/`

The script handles variable input image sizes by resizing to 224×224 with torchvision transforms and saves checkpoints using the requested names `maturity_best-3-1.pth` and `quality_best-3-1.pth`.

## Dataset layout
```
<root>
├── Maturity/
│   ├── Immature/
│   └── Mature/
└── Quality/
    ├── Fresh/
    └── Rotten/
```

Place your images in the matching class folders. The script automatically splits each task into train/validation sets.

## Training
Train both maturity and quality models (ResNet-18 fine-tuned with ImageNet weights by default):
```bash
python ml/train_fruit_models.py train --data-root /path/to/data --epochs 15 --batch-size 32 --output-dir checkpoints
```

Train only one task:
```bash
python ml/train_fruit_models.py train --data-root /path/to/data --task maturity
python ml/train_fruit_models.py train --data-root /path/to/data --task quality
```

The best validation checkpoints are saved to `maturity_best-3-1.pth` and `quality_best-3-1.pth` inside `--output-dir`.

## Prediction
Run inference on a single image (device auto-selects CUDA if available during training; you can override for prediction):
```bash
python ml/train_fruit_models.py predict --checkpoint checkpoints/maturity_best-3-1.pth --image /path/to/image.jpg --device cpu
```

The command prints the predicted class and per-class probabilities.
