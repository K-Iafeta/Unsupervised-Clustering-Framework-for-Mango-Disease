#!/usr/bin/env python
# coding: utf-8

# # Experiment 2: Fine-Tuned Feature Extractor
# 
# - Section 0-4, Feature Extraction -> embeddings and classification
# - Section 5-7, unsupervised clustering with feature embeddings
# - Section 8-10, Cluster-aware inference -> embeddings and classification
# 

# # Section 0-4, Feature Extraction Methodology
# ## **Feature Extraction 1 - Base ViT-B/16**

# ## 0. Imports and Configurations

# In[1]:


### IMPORTS ###

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split

# pip install transformers
from transformers import ViTModel, ViTImageProcessor

### Reproducibility Variables ###

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

### Device ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Configuration

CSV_PATH = "MangoImageSample.csv"
MODEL_NAME = "google/vit-base-patch16-224"
EMBEDDING_DIM = 256
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
OUTPUT_FOLDER = "Saved_Embeddings"
MODEL_FOLDER = "Saved_Models"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

print("Configuration loaded.")

# 1. PREPARING DATASET
# Using the SAME train/validation/test split as Experiment 1
### 1.1 Load Existing Metadata Splits ###

TRAIN_CSV = "train_metadata.csv"
VAL_CSV = "validation_metadata.csv"
TEST_CSV = "test_metadata.csv"

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Existing metadata splits loaded.")

print("\nTraining shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)


### 1.2 Check Columns ###

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nValidation columns:")
print(val_df.columns.tolist())

print("\nTest columns:")
print(test_df.columns.tolist())


# Columns required by the feature-extraction pipeline

required_columns = [
    "path",
    "Category",
    "Class",
    "Plant_Part",
    "class_id"
]

for name, dataframe in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    missing = [
        col for col in required_columns
        if col not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{name} metadata is missing columns: {missing}"
        )

print("\nAll required columns are present.")


### 1.3 Reset Indices ###

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


### 1.4 Dataset Sizes ###

print("\nDataset sizes:")

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

total_images = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print("Total:", total_images)


### 1.5 Use Existing Class IDs ###

# DO NOT recreate class_id.
# These IDs already come from Experiment 1.

class_names = (
    train_df[["class_id", "Class"]]
    .drop_duplicates()
    .sort_values("class_id")
)

print("\nExisting class mapping:")

for _, row in class_names.iterrows():

    print(
        int(row["class_id"]),
        "->",
        row["Class"]
    )


# Create dictionaries from the existing metadata

class_to_idx = dict(
    zip(
        class_names["Class"],
        class_names["class_id"]
    )
)

idx_to_class = dict(
    zip(
        class_names["class_id"],
        class_names["Class"]
    )
)

NUM_CLASSES = len(class_names)

print("\nNumber of disease classes:", NUM_CLASSES)


### 1.6 Verify Class IDs Across All Splits ###

print("\nClass IDs in training:")
print(sorted(train_df["class_id"].unique()))

print("\nClass IDs in validation:")
print(sorted(val_df["class_id"].unique()))

print("\nClass IDs in test:")
print(sorted(test_df["class_id"].unique()))


# Check that every class_id has the same Class name
# across all three datasets.

for dataframe, name in [
    (train_df, "Training"),
    (val_df, "Validation"),
    (test_df, "Test")
]:

    mapping_check = (
        dataframe[["class_id", "Class"]]
        .drop_duplicates()
        .sort_values("class_id")
    )

    for _, row in mapping_check.iterrows():

        class_id = row["class_id"]
        class_name = row["Class"]

        if class_id not in idx_to_class:
            raise ValueError(
                f"{name}: unknown class_id {class_id}"
            )

        if idx_to_class[class_id] != class_name:
            raise ValueError(
                f"{name}: class_id {class_id} "
                f"has inconsistent class name "
                f"'{class_name}'"
            )

print("\nClass ID consistency verified.")


### 1.7 Class Distributions ###

print("\nTRAINING CLASS DISTRIBUTION:")
print(train_df["Class"].value_counts())

print("\nVALIDATION CLASS DISTRIBUTION:")
print(val_df["Class"].value_counts())

print("\nTEST CLASS DISTRIBUTION:")
print(test_df["Class"].value_counts())


### 1.8 Class Proportions ###

print("\nTRAINING CLASS PROPORTIONS:")
print(train_df["Class"].value_counts(normalize=True))

print("\nVALIDATION CLASS PROPORTIONS:")
print(val_df["Class"].value_counts(normalize=True))

print("\nTEST CLASS PROPORTIONS:")
print(test_df["Class"].value_counts(normalize=True))

# 1.9 CREATE PYTORCH DATASET
class MangoDataset(Dataset):

    def __init__(self, dataframe, processor):

        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["path"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # ViT preprocessing
        processed = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = processed["pixel_values"].squeeze(0)

        # Use EXISTING class_id from metadata
        label = torch.tensor(
            row["class_id"],
            dtype=torch.long
        )

        return {
            "image": pixel_values,
            "class": label,
            "class_name": row["Class"],
            "category": row["Category"],
            "plant_part": row["Plant_Part"],
            "path": image_path
        }

# ============================================================
# 1.10 LOAD ViT-B/16 PROCESSOR
# ============================================================

processor = ViTImageProcessor.from_pretrained(
    MODEL_NAME
)

print("\nViT processor loaded.")


# 1.11 CREATE DATASETS
train_dataset = MangoDataset(
    train_df,
    processor
)

val_dataset = MangoDataset(
    val_df,
    processor
)

test_dataset = MangoDataset(
    test_df,
    processor
)

print("\nDatasets created.")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# 1.12 CREATE DATALOADERS

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("DataLoaders created.")

# ## 2. Feature Extraction Architecture

# In[7]:


# 2.1 Fine-Tuned VIT-B/16 EMBEDDING + CLASSIFIER
class FineTunedViT(nn.Module):
    def __init__(
        self,
        model_name=MODEL_NAME,
        embedding_dim=256,
        num_classes=10
    ):
        super().__init__()

        self.vit = ViTModel.from_pretrained(model_name)
        hidden_size = self.vit.config.hidden_size
        self.embedding_layer = nn.Linear(hidden_size,embedding_dim)
        self.classifier = nn.Linear(embedding_dim,num_classes)

    def forward(
        self,
        pixel_values,
        return_embeddings=False
    ):

        outputs = self.vit(
            pixel_values=pixel_values
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        embeddings = self.embedding_layer(cls_output)

        if return_embeddings:
            return embeddings
        logits = self.classifier(embeddings)
        return logits

# 2.2 INITIALIZE MODEL # 
model = FineTunedViT(
    model_name=MODEL_NAME,
    embedding_dim=EMBEDDING_DIM,
    num_classes=NUM_CLASSES
).to(device)
print(model)

# 2.3 Verify Trainable Parameters -> ViT frozen #
# Trainable params a lot less than total means frozen #
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad]

total_trainable = sum(
    p.numel()
    for p in trainable_params)

total_params = sum(
    p.numel()
    for p in model.parameters())

print(f"Total parameters: "f"{total_params:,}")
print(f"Trainable parameters: "f"{total_trainable:,}")


# ## 3. Train Supervised Classifier
# xi -> zi -> y^(i)

# In[23]:


# 3.1 Loss and Optimizer #

criterion = nn.CrossEntropyLoss()
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad]

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.vit.parameters(), "lr": 1e-5
        },
        {
            "params": model.embedding_layer.parameters(),"lr": 1e-3
        },
        {
            "params": model.classifier.parameters(),"lr": 1e-3
        }
    ],
    weight_decay=0.01
)
print("Optimizer ready.")

# 3.2 TRAINING FUNCTION #

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["class"].to(device)

        # Forward #
        logits = model(pixel_values=images)

        # Loss #
        loss = criterion(logits,labels)

        # Backpropagation #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = (running_loss / len(loader))
    epoch_accuracy = (correct / total)
    return epoch_loss, epoch_accuracy

# 3.3 VALIDATION FUNCTION #

@torch.no_grad()
def evaluate(model,loader,criterion,device):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)

        labels = batch["class"].to(device)

        logits = model(pixel_values=images)

        loss = criterion(logits,labels)
        running_loss += loss.item()
        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = (running_loss / len(loader))
    epoch_accuracy = (correct / total)

    return epoch_loss, epoch_accuracy


# 3.4 TRAINING LOOP #

best_val_accuracy = 0.0
history = []
best_model_path = os.path.join(MODEL_FOLDER,"best_fine_vit_embedding_model.pth")

for epoch in range(NUM_EPOCHS):
    # Training
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device)

    # Validation #
    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device)

    # Save history #
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy})
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            best_model_path)
        print(
            "  -> Best model saved."
        )
print("\nBest validation accuracy:",best_val_accuracy)
print("Model saved to:",best_model_path)


# In[ ]:


# 3.5 Training History #

history_df = pd.DataFrame(history)
print(history_df)

# 3.6 Load Best Model #
model.load_state_dict(torch.load(best_model_path,map_location=device))
model.to(device)
model.eval()
print("Best model loaded.")


# ## 4. Extract Penultimate (zi) Embeddings (Representation Analysis)

# In[41]:


### 4.1 EMBEDDING EXTRACTION ###
@torch.no_grad()
def extract_embeddings(model,loader,device):
    model.eval()

    all_embeddings = []
    all_labels = []
    all_class_names = []
    all_categories = []
    all_plant_parts = []
    all_paths = []

    for batch in loader:
        images = batch["image"].to(device)

        # Extract penultimate representation #

        embeddings = model(pixel_values=images,return_embeddings=True)

        # Move to CPU #
        all_embeddings.append(embeddings.cpu())

        # Metadata #
        all_labels.append(batch["class"])

        all_class_names.extend(batch["class_name"])

        all_categories.extend(batch["category"])

        all_plant_parts.extend(batch["plant_part"])

        all_paths.extend(batch["path"])

    # Combine batches #

    embeddings = torch.cat(all_embeddings,dim=0).numpy()
    labels = torch.cat(all_labels,dim=0).numpy()
    return {
        "embeddings": embeddings,
        "class_id": labels,
        "class_name": np.array(
            all_class_names),
        "category": np.array(
            all_categories),
        "plant_part": np.array(
            all_plant_parts),
        "path": np.array(
            all_paths
        )
    }

# 4.2 Training Embeddings (For Clustering) #

train_embeddings = extract_embeddings(model,train_loader,device)
print("Training embedding shape:",train_embeddings["embeddings"].shape)

# 4.3 Validation Embeddings #

val_embeddings = extract_embeddings(model,val_loader,device)
print("Validation embedding shape:",val_embeddings["embeddings"].shape)

# 4.4 TEST EMBEDDINGS #

test_embeddings = extract_embeddings(model,test_loader,device)
print("Test embedding shape:",test_embeddings["embeddings"].shape)
print(" ")

# 4.5 Inspect Embeddings #
cluster_emb = train_embeddings["embeddings"]

print("Embedding dimensions:",cluster_emb.shape[1])
print("Number of training embeddings:",cluster_emb.shape[0])
print("Example embedding:")
print(cluster_emb[0])

# 4.6 Save Training Embeddings #
#  old name: "FT_train_fine_vit_embeddings.npz")

train_npz_path = os.path.join(OUTPUT_FOLDER,"train_fine_vit_embeddings.npz")

np.savez_compressed(
    train_npz_path,
    embeddings=train_embeddings["embeddings"],
    class_id=train_embeddings["class_id"],
    class_name=train_embeddings["class_name"],
    category=train_embeddings["category"],
    plant_part=train_embeddings["plant_part"],
    path=train_embeddings["path"])

print("Training embeddings saved to:",train_npz_path)

# 4.7 Save Validation Embeddings #

# old name: "FT_validation_fine_vit_embeddings.npz")
val_npz_path = os.path.join(OUTPUT_FOLDER,"validation_fine_vit_embeddings.npz")
np.savez_compressed(
    val_npz_path,
    embeddings=val_embeddings["embeddings"],
    class_id=val_embeddings["class_id"],
    class_name=val_embeddings["class_name"],
    category=val_embeddings["category"],
    plant_part=val_embeddings["plant_part"],
    path=val_embeddings["path"])

print("Validation embeddings saved to:",val_npz_path)

# 4.8 Save Test Embeddings #

test_npz_path = os.path.join(OUTPUT_FOLDER,"test_fine_vit_embeddings.npz")
np.savez_compressed(
    test_npz_path,
    embeddings=test_embeddings["embeddings"],
    class_id=test_embeddings["class_id"],
    class_name=test_embeddings["class_name"],
    category=test_embeddings["category"],
    plant_part=test_embeddings["plant_part"],
    path=test_embeddings["path"])

print("Test embeddings saved to:",test_npz_path)

# 4.9 Save Training Embeddings as .csv #

embedding_columns = [
    f"Embedding_{i}"
    for i in range(EMBEDDING_DIM)]

embedding_df = pd.DataFrame(train_embeddings["embeddings"],columns=embedding_columns)
embedding_df["Class"] = (train_embeddings["class_name"])
embedding_df["Category"] = (train_embeddings["category"])
embedding_df["Plant_Part"] = (train_embeddings["plant_part"])
embedding_df["path"] = (train_embeddings["path"])

embedding_csv_path = os.path.join(OUTPUT_FOLDER,"train_fine_vit_embeddings.csv")
embedding_df.to_csv(embedding_csv_path,index=False)

print("Training embedding CSV saved to:",embedding_csv_path)

# 4.10 VERIFY EMBEDDING / METADATA ALIGNMENT#

n_embeddings = len(train_embeddings["embeddings"])
n_paths = len(train_embeddings["path"])
n_classes = len(train_embeddings["class_name"])
n_categories = len(train_embeddings["category"])
n_parts = len(train_embeddings["plant_part"])

print("Embeddings:", n_embeddings)
print("Paths:", n_paths)
print("Classes:", n_classes)
print("Categories:", n_categories)
print("Plant parts:", n_parts)

assert n_embeddings == n_paths
assert n_embeddings == n_classes
assert n_embeddings == n_categories
assert n_embeddings == n_parts

print("\nAlignment check passed.")