# PointNet-Based Pedicle Landmark and Segmentation Pipeline

This repository provides a full pipeline for point cloud-based vertebral landmark detection and segmentation using a PointNet-based neural network architecture. It includes preprocessing, augmentation, training, and evaluation of a model tailored to spinal anatomy datasets.

---

## 📁 Repository Structure

├── dataset.py              # Custom PyTorch Dataset class for loading NPZ point clouds
├── model.py                # PointNet model definitions (segmentation, classification)
├── trainer.py              # Training script for segmentation using PointNet
├── preprocessing.py        # Mesh augmentation, landmark labeling, noise addition
├── folding.py              # Generates train/val/test splits from NPZ files
├── data/                   # Your structured dataset directory (STLs + NPZ files)
│   ├── fold_1/
│   │   ├── train_data.json
│   │   ├── val_data.json
│   │   ├── test_data.json

---

## 🧰 Requirements

Install dependencies with:

```bash
pip install torch numpy pyvista scikit-learn tqdm

Tested with:
	•	Python 3.8+
	•	PyTorch ≥ 1.11
	•	PyVista for STL mesh loading and manipulation

⸻

🚀 Usage

1. Preprocess STL Meshes and Landmarks

Prepare your .stl meshes and .npz landmark files, then run:

python preprocessing.py

This will:
	•	Augment meshes (if "USR" in filename)
	•	Add Gaussian noise to vertices
	•	Compute label heatmaps
	•	Save .npz point clouds and augmented .stl files

2. Create Dataset Splits

python folding.py

Creates:
	•	train_data.json
	•	val_data.json
	•	test_data.json
inside fold_1/, ready for training.

3. Train the Model

python trainer.py --dataset path/to/dataset --class_choice spine

Optional flags:
	•	--feature_transform — enables feature transform regularization
	•	--model path/to/model.pth — resumes training from a checkpoint

⸻

📦 NPZ File Format

Each .npz file contains:
	•	vertices: (N, 3) point cloud
	•	labels: (N,) semantic class labels (0 = background, 1–4 = landmark types)
	•	landmarks: (4, 3) landmark coordinates

⸻

📈 Output

After training:
	•	Model checkpoints saved to seg_noisy/
	•	Training/test accuracy printed per epoch
	•	Final mIoU score reported at end





