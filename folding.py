import os
import json
from sklearn.model_selection import train_test_split

# Directories
dir = "../new_new_dataset"
save_dir = os.path.join(dir,"fold_1")

# Create save directory if not exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



# Separate files based on "Specimen_"
test_files = []
train_files = []

# Get file list
for file in os.listdir(dir):
    if not file.endswith(".npz"):
        continue
    if "Specimen" in file:
        test_files.append("pedicle_dataset/1/" + file[:-4])
    else:
        train_files.append("pedicle_dataset/1/" + file[:-4])

# Split train_files into 80% train and 20% validation
train_split, val_split = train_test_split(train_files, test_size=0.2, random_state=42)

# Save train, validation, and test data
with open(os.path.join(save_dir, 'train_data.json'), 'w') as f:
    json.dump(train_split, f)
with open(os.path.join(save_dir, 'val_data.json'), 'w') as f:
    json.dump(val_split, f)
with open(os.path.join(save_dir, 'test_data.json'), 'w') as f:
    json.dump(test_files, f)

# Print summary
print(f"Train set: {len(train_split)} files")
print(f"Validation set: {len(val_split)} files")
print(f"Test set: {len(test_files)} files")
