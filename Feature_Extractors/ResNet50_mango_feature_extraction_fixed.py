#!/usr/bin/env python
# coding: utf-8

# # Experiment 3: ResNet -> Fine-Tuned Feature Extractor
# 
# - Section 0-4, Feature Extraction -> ResNet-50 embeddings and classification
# - Section 5-7, unsupervised clustering with feature embeddings
# - Section 8-10, Cluster-aware inference -> ResNet-50 embeddings and classification
# 

# # Section 0-4, Feature Extraction Methodology
# ## **Experiment 3 - ResNet-50 Fine-Tuned Feature Extractor**

# ## 0. Imports and Configurations

# In[3]:


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
from torchvision.models import resnet50, ResNet50_Weights
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
MODEL_NAME = "resnet50"
EMBEDDING_DIM = 256
BATCH_SIZE = 32
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
OUTPUT_FOLDER = "Saved_Embeddings"
MODEL_FOLDER = "Saved_Models"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

print("Configuration loaded.")


# ## 1. Preparing Dataset
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

### 1.8 Create PyTorch Dataset ###
class MangoDataset(Dataset):

    def __init__(self,dataframe,transform,class_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        image_path = row["path"]

        # Load image #
        image = Image.open(image_path).convert("RGB")

        ## ResNet-50 preprocessing ##
        pixel_values = self.transform(image)

        # Disease label
        label = torch.tensor(self.class_to_idx[row["Class"]],dtype=torch.long)
        return {
                "image": pixel_values,
                "class": label,
                "class_name": row["Class"],
                "category": row["Category"],
                "plant_part": row["Plant_Part"],
                "path": image_path}

# 1.9 Load ResNet-50 preprocessing
# Use the preprocessing associated with the pretrained ResNet-50 weights.
resnet_weights = ResNet50_Weights.DEFAULT
transform = resnet_weights.transforms()
print("ResNet-50 preprocessing loaded.")

# 1.10 CREATE DATASETS #
train_dataset = MangoDataset(train_df,transform,class_to_idx)
val_dataset = MangoDataset(val_df,transform,class_to_idx)
test_dataset = MangoDataset(test_df,transform,class_to_idx)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# 1.11 DataLoader #
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
print("DataLoaders created.")


# In[6]:


## 1.12 Checking for bad images ## 
from PIL import Image
import pandas as pd

metadata = pd.read_csv(CSV_PATH)

bad_images = []

for path in metadata["path"]:

    try:
        with Image.open(path) as img:
            img.verify()

    except Exception as e:
        bad_images.append({
            "path": path,
            "error": str(e)
        })

print(f"Total images: {len(metadata)}")
print(f"Problematic images: {len(bad_images)}")

bad_df = pd.DataFrame(bad_images)
bad_df


# ## 2. Feature Extraction Architecture

# In[12]:


# 2.1 ResNet-50 EMBEDDING + CLASSIFIER

class ResNet50EmbeddingModel(nn.Module):
    """
    ResNet-50 backbone -> 2048-D pooled visual features
    -> 256-D learned embedding -> disease classifier.

    The 256-D embedding is the representation used for clustering.
    The classifier is trained from the same embedding and can therefore
    be reused during cluster-aware inference.
    """

    def __init__(self, embedding_dim=256, num_classes=10):
        super().__init__()

        # Pretrained ImageNet ResNet-50.
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Replace the ImageNet classifier with Identity so that the
        # forward pass returns the 2048-D global-average-pooled features.
        self.resnet.fc = nn.Identity()

        # Learned projection into the clustering/classification space.
        self.embedding_layer = nn.Linear(2048, embedding_dim)

        # Final supervised disease classifier.
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, pixel_values, return_embeddings=False):
        # [B, 3, 224, 224] -> [B, 2048]
        features = self.resnet(pixel_values)

        # [B, 2048] -> [B, 256]
        embeddings = self.embedding_layer(features)

        if return_embeddings:
            return embeddings

        # [B, 256] -> [B, NUM_CLASSES]
        logits = self.classifier(embeddings)

        return logits


# 2.2 INITIALIZE MODEL
model = ResNet50EmbeddingModel(
    embedding_dim=EMBEDDING_DIM,
    num_classes=NUM_CLASSES
).to(device)

print(model)

# 2.3 Verify Trainable Parameters
# The ResNet backbone is fine-tuned in this experiment.
for p in model.parameters():
    p.requires_grad = True
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
# x_i -> z_i -> y_hat_i

# In[23]:


# 3.1 Loss and Optimizer #

criterion = nn.CrossEntropyLoss()
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad]

optimizer = torch.optim.AdamW(
    [
        {"params": model.resnet.parameters(), "lr": 1e-4},
        {"params": model.embedding_layer.parameters(), "lr": LEARNING_RATE},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
    ],
    weight_decay=WEIGHT_DECAY
)
print("Optimizer ready.")
print("ResNet-50 backbone LR: 1e-4")
print("Embedding/classifier LR:", LEARNING_RATE)

# 3.2 TRAINING FUNCTION #

import time

def train_one_epoch(model,loader,criterion,optimizer,device):

    model.train()

    running_loss=0.0
    correct=0
    total=0

    epoch_start=time.time()

    for batch_idx,batch in enumerate(loader):

        batch_start=time.time()

        images=batch["image"].to(device,non_blocking=True)
        labels=batch["class"].to(device,non_blocking=True)

        # Forward
        logits=model(pixel_values=images)

        # Loss
        loss=criterion(logits,labels)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss+=loss.item()

        predictions=logits.argmax(dim=1)
        correct+=(predictions==labels).sum().item()
        total+=labels.size(0)

        batch_time=time.time()-batch_start

        if batch_idx%10==0:
            print(
                f"Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {batch_time:.2f}s",
                flush=True
            )

    epoch_loss=running_loss/len(loader)
    epoch_accuracy=correct/total

    print(
        f"Epoch completed | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy:.4f} | "
        f"Time: {(time.time()-epoch_start)/60:.2f} min",
        flush=True
    )

    return epoch_loss,epoch_accuracy

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
best_model_path = os.path.join(MODEL_FOLDER,"best_resnet50_embedding_model.pth")

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

# Sanity check: confirm the embedding has the intended dimensionality.
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    sample_images = sample_batch["image"].to(device)
    sample_embeddings = model(sample_images, return_embeddings=True)

print("Sanity-check embedding shape:", tuple(sample_embeddings.shape))
assert sample_embeddings.shape[1] == EMBEDDING_DIM
print(f"Embedding dimension confirmed: {EMBEDDING_DIM}")


# ## 4. Extract Penultimate (zi) Embeddings (Representation Analysis)

# In[41]:


### 4.1 RESNET-50 EMBEDDING EXTRACTION ###
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

# These are the exact 256-D learned representations used for clustering.
# Apply StandardScaler later in the clustering stage, and save/reuse that
# same scaler for cluster-aware inference.
print("Example embedding:")
print(cluster_emb[0])

# 4.6 Save Training Embeddings #

train_npz_path = os.path.join(OUTPUT_FOLDER,"train_resnet50_embeddings.npz")

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

val_npz_path = os.path.join(OUTPUT_FOLDER,"validation_resnet50_embeddings.npz")
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

test_npz_path = os.path.join(OUTPUT_FOLDER,"test_resnet50_embeddings.npz")
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

embedding_csv_path = os.path.join(OUTPUT_FOLDER,"train_resnet50_embeddings.csv")
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
