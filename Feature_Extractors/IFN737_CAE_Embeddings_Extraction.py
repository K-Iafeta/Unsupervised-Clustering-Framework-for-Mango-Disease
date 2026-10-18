#!/usr/bin/env python
# coding: utf-8

# # Experiment 4: CAE -> Fine-Tuned Feature Extractor
# 
# Stage 1: Image -> CAE Encoder -> 256-D -> Decoder
# 
# Stage 2: Image -> pretrained Encoder -> 256-D -> Classifier
# 
# Final: 256-D embeddings -> StandardScaler -> clustering
# 
# 
# - Section 0-4, Feature Extraction -> embeddings and classification
# - Section 5-7, unsupervised clustering with feature embeddings
# - Section 8-10, Cluster-aware inference -> embeddings and classification
# 

# # Section 0-4, Feature Extraction Methodology
# ## **Feature Extraction 1 - Base ViT-B/16**

# ## 0. Imports and Configurations

# In[2]:


import os,random,numpy as np,pandas as pd
from PIL import Image,ImageFile
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

SEED=42
IMAGE_SIZE=224
EMBEDDING_DIM=256
BATCH_SIZE=32
CAE_NUM_EPOCHS=2
CAE_LEARNING_RATE=1e-3
CAE_WEIGHT_DECAY=0.01
FINETUNE_NUM_EPOCHS=2
FINETUNE_LEARNING_RATE=1e-4
CLASSIFIER_LEARNING_RATE=1e-3
FINETUNE_WEIGHT_DECAY=0.01
OUTPUT_FOLDER="Saved_Embeddings"
MODEL_FOLDER="Saved_Models"

os.makedirs(OUTPUT_FOLDER,exist_ok=True)
os.makedirs(MODEL_FOLDER,exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)
if torch.cuda.is_available():print("GPU:",torch.cuda.get_device_name(0))
ImageFile.LOAD_TRUNCATED_IMAGES=True


# ## 1. Preparing Dataset

# In[3]:


# 1.1 LOAD TRAIN / VALIDATION / TEST METADATA

TRAIN_CSV="train_metadata.csv"
VAL_CSV="validation_metadata.csv"
TEST_CSV="test_metadata.csv"

train_df=pd.read_csv(TRAIN_CSV)
val_df=pd.read_csv(VAL_CSV)
test_df=pd.read_csv(TEST_CSV)

print("Train:",train_df.shape)
print("Validation:",val_df.shape)
print("Test:",test_df.shape)

required_columns=["path","Class","class_id","Plant_Part", "Category"]

for df,name in [(train_df,"train"),(val_df,"validation"),(test_df,"test")]:
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' from {name} metadata.")

print("\nRequired metadata columns found.")

NUM_CLASSES=train_df["class_id"].nunique()

print("\nNumber of classes:",NUM_CLASSES)

print("\nTraining class distribution:")
print(train_df["class_id"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["class_id"].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df["class_id"].value_counts().sort_index())


# 1.2 IMAGE PREPROCESSING

cae_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor()
])


# 1.3 MAKING DATASET

class MangoDataset(Dataset):

    def __init__(self,dataframe,transform=None,return_labels=True):
        self.dataframe=dataframe.reset_index(drop=True)
        self.transform=transform
        self.return_labels=return_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        row=self.dataframe.iloc[idx]

        image=Image.open(row["path"]).convert("RGB")

        if self.transform:
            image=self.transform(image)

        if self.return_labels:
            return image,int(row["class_id"])
        else:
            return image


train_dataset=MangoDataset(
    train_df,
    cae_transform,
    True
)

val_dataset=MangoDataset(
    val_df,
    cae_transform,
    True
)

test_dataset=MangoDataset(
    test_df,
    cae_transform,
    True
)


# 1.4 DATALOADERS

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("\nDataLoaders created.")
print("Training batches:",len(train_loader))
print("Validation batches:",len(val_loader))
print("Test batches:",len(test_loader))


# 1.6 CHECK FOR BAD / UNREADABLE IMAGES

def check_images(dataframe):

    bad_images=[]

    for path in dataframe["path"]:

        try:
            with Image.open(path) as img:
                img.verify()

        except Exception as e:
            bad_images.append((path,str(e)))

    return bad_images


bad_train=check_images(train_df)
bad_val=check_images(val_df)
bad_test=check_images(test_df)

print("\nBad training images:",len(bad_train))
print("Bad validation images:",len(bad_val))
print("Bad test images:",len(bad_test))


# ## 2. Feature Extraction Architecture

# In[4]:


print("Train:",train_df.columns.tolist())
print("Validation:",val_df.columns.tolist())
print("Test:",test_df.columns.tolist())


# In[5]:


### 2.1 Define CAE Models ###
# CAE Encoder
class CAEEncoder(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.embedding_layer = nn.Linear(
            512 * 7 * 7,
            embedding_dim
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.embedding_layer(torch.flatten(x, 1))
        return x

# CAE Decoder
class CAEDecoder(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.decoder_input = nn.Linear(
            embedding_dim,
            512 * 7 * 7
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding):
        x = self.decoder_input(embedding)
        x = x.view(-1, 512, 7, 7)
        x = self.decoder(x)
        return x

# Complete CAE
class ConvolutionalAutoencoder(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.encoder = CAEEncoder(embedding_dim)
        self.decoder = CAEDecoder(embedding_dim)

    def forward(self, x, return_embeddings=False):
        embedding = self.encoder(x)

        if return_embeddings:
            return embedding

        reconstruction = self.decoder(embedding)

        return reconstruction


# Create CAE model
cae_model = ConvolutionalAutoencoder(EMBEDDING_DIM).to(device)

print("\nCAE model created.")
print("Number of parameters:",sum(p.numel() for p in cae_model.parameters()))


# In[6]:


### 2.2 Stage One: CAE Pretraining ###

reconstruction_loss_fn = nn.MSELoss()

cae_optimizer = torch.optim.AdamW(
    cae_model.parameters(),
    lr=CAE_LEARNING_RATE,
    weight_decay=CAE_WEIGHT_DECAY
)


def train_cae_epoch(model, loader, optimizer, loss_function):

    model.train()

    running_loss = 0.0
    total_images = 0

    for images, _ in loader:

        images = images.to(device)

        optimizer.zero_grad()

        reconstructions = model(images)

        loss = loss_function(
            reconstructions,
            images
        )

        loss.backward()

        optimizer.step()

        n = images.size(0)

        running_loss += loss.item() * n
        total_images += n

    return running_loss / total_images


def validate_cae(model, loader, loss_function):

    model.eval()

    running_loss = 0.0
    total_images = 0

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            reconstructions = model(images)

            loss = loss_function(
                reconstructions,
                images
            )

            n = images.size(0)

            running_loss += loss.item() * n
            total_images += n

    return running_loss / total_images


# Train CAE

best_cae_val_loss = float("inf")

cae_history = []

best_cae_path = os.path.join(
    MODEL_FOLDER,
    "best_cae_pretrained_model.pth"
)


print("\n========================================")
print("STAGE 1: CAE PRETRAINING")
print("========================================")


for epoch in range(CAE_NUM_EPOCHS):

    train_loss = train_cae_epoch(
        cae_model,
        train_loader,
        cae_optimizer,
        reconstruction_loss_fn
    )

    val_loss = validate_cae(
        cae_model,
        val_loader,
        reconstruction_loss_fn
    )

    cae_history.append({
        "epoch": epoch + 1,
        "train_reconstruction_loss": train_loss,
        "validation_reconstruction_loss": val_loss
    })

    print(
        f"Epoch {epoch + 1}/{CAE_NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

    if val_loss < best_cae_val_loss:

        best_cae_val_loss = val_loss

        torch.save(
            cae_model.state_dict(),
            best_cae_path
        )

        print("  -> Saved best CAE model.")


# Load best CAE model

cae_model.load_state_dict(
    torch.load(
        best_cae_path,
        map_location=device
    )
)

print(
    "\nBest CAE validation reconstruction loss:",
    best_cae_val_loss
)


# 2.3 Stage Two: Supervised Fine-tuning


class FineTunedCAE(nn.Module):

    def __init__(
        self,
        pretrained_encoder,
        embedding_dim=256,
        num_classes=10
    ):
        super().__init__()

        self.encoder = pretrained_encoder

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )


    def forward(
        self,
        x,
        return_embeddings=False
    ):

        embedding = self.encoder(x)

        if return_embeddings:
            return embedding

        return self.classifier(embedding)


# Create fine-tuned CAE model

finetuned_cae_model = FineTunedCAE(
    cae_model.encoder,
    EMBEDDING_DIM,
    NUM_CLASSES
).to(device)


print("\nFine-tuned CAE model created.")


print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in finetuned_cae_model.parameters()
        if p.requires_grad
    )
)


# ## 3. Train Supervised Classifier
# xi -> zi -> y^(i)

# In[7]:


# 3.1 Classification Loss and Optimizer

classification_loss_fn = nn.CrossEntropyLoss()

finetune_optimizer = torch.optim.AdamW(
    [
        {
            "params": finetuned_cae_model.encoder.parameters(),
            "lr": FINETUNE_LEARNING_RATE
        },
        {
            "params": finetuned_cae_model.classifier.parameters(),
            "lr": CLASSIFIER_LEARNING_RATE
        }
    ],
    weight_decay=FINETUNE_WEIGHT_DECAY
)


# 3.2 Supervised Training

def train_classifier_epoch(
    model,
    loader,
    optimizer,
    loss_function
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_function(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        n = images.size(0)

        running_loss += loss.item() * n

        correct += (
            logits.argmax(1) == labels
        ).sum().item()

        total += n

    return (
        running_loss / total,
        correct / total
    )


# 3.3 Validation

def validate_classifier(
    model,
    loader,
    loss_function
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = loss_function(
                logits,
                labels
            )

            n = images.size(0)

            running_loss += loss.item() * n

            correct += (
                logits.argmax(1) == labels
            ).sum().item()

            total += n

    return (
        running_loss / total,
        correct / total
    )


# 3.4 Train Fine-Tuned CAE Classifier

best_val_accuracy = 0.0

finetune_history = []

best_finetuned_cae_path = os.path.join(
    MODEL_FOLDER,
    "best_finetuned_cae_embedding_model.pth"
)


print("\n========================================")
print("STAGE 2: SUPERVISED CAE FINE-TUNING")
print("========================================")


for epoch in range(FINETUNE_NUM_EPOCHS):

    train_loss, train_accuracy = train_classifier_epoch(
        finetuned_cae_model,
        train_loader,
        finetune_optimizer,
        classification_loss_fn
    )

    val_loss, val_accuracy = validate_classifier(
        finetuned_cae_model,
        val_loader,
        classification_loss_fn
    )

    finetune_history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy
        }
    )

    print(
        f"Epoch {epoch + 1}/{FINETUNE_NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            finetuned_cae_model.state_dict(),
            best_finetuned_cae_path
        )

        print("  -> Saved best fine-tuned CAE model.")


# Load best fine-tuned CAE model

finetuned_cae_model.load_state_dict(
    torch.load(
        best_finetuned_cae_path,
        map_location=device
    )
)

print(
    "\nBest validation classification accuracy:",
    best_val_accuracy
)


# 3.5 Final Test Performance

test_loss, test_accuracy = validate_classifier(
    finetuned_cae_model,
    test_loader,
    classification_loss_fn
)


print("\n========================================")
print("FINAL TEST PERFORMANCE")
print("========================================")

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# ## 4. Extract Penultimate (zi) Embeddings (Representation Analysis)

# In[10]:


# ============================================================
# 4. EXTRACT PENULTIMATE (z_i) EMBEDDINGS
# ============================================================

# ------------------------------------------------------------
# 4.1 Create extraction DataLoaders
# ------------------------------------------------------------
#
# IMPORTANT:
# The training DataLoader used during training has shuffle=True.
#
# We MUST use shuffle=False for embedding extraction so that
# the order of the embeddings matches the order of the
# corresponding metadata rows.
# ------------------------------------------------------------

embedding_train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

embedding_val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

embedding_test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ------------------------------------------------------------
# 4.2 Extract Final 256-D Embeddings
# ------------------------------------------------------------

def extract_embeddings(
    model,
    loader
):

    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            embeddings = model(
                images,
                return_embeddings=True
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )


    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )


    return embeddings, labels


# ------------------------------------------------------------
# 4.3 Extract Train / Validation / Test Embeddings
# ------------------------------------------------------------

train_embeddings, train_labels = extract_embeddings(
    finetuned_cae_model,
    embedding_train_loader
)

validation_embeddings, validation_labels = extract_embeddings(
    finetuned_cae_model,
    embedding_val_loader
)

test_embeddings, test_labels = extract_embeddings(
    finetuned_cae_model,
    embedding_test_loader
)


print("\nEmbedding shapes:")

print(
    "Train:",
    train_embeddings.shape
)

print(
    "Validation:",
    validation_embeddings.shape
)

print(
    "Test:",
    test_embeddings.shape
)


# ------------------------------------------------------------
# 4.4 Verify Embedding Dimension
# ------------------------------------------------------------

assert train_embeddings.shape[1] == EMBEDDING_DIM
assert validation_embeddings.shape[1] == EMBEDDING_DIM
assert test_embeddings.shape[1] == EMBEDDING_DIM


print(
    "\nAll embeddings have dimension:",
    EMBEDDING_DIM
)


# ------------------------------------------------------------
# 4.5 Verify Class IDs
# ------------------------------------------------------------

assert np.array_equal(
    train_labels,
    train_df["class_id"].values
)

assert np.array_equal(
    validation_labels,
    val_df["class_id"].values
)

assert np.array_equal(
    test_labels,
    test_df["class_id"].values
)


print(
    "\nClass ID / embedding ordering verified."
)


# ============================================================
# 4.6 SAVE EMBEDDINGS AS NPZ
# ============================================================

train_npz_path = os.path.join(
    OUTPUT_FOLDER,
    "train_cae_embeddings.npz"
)

validation_npz_path = os.path.join(
    OUTPUT_FOLDER,
    "validation_cae_embeddings.npz"
)

test_npz_path = os.path.join(
    OUTPUT_FOLDER,
    "test_cae_embeddings.npz"
)


# ------------------------------------------------------------
# Training NPZ
# ------------------------------------------------------------

np.savez_compressed(
    train_npz_path,
    embeddings=train_embeddings,
    class_id=train_df["class_id"].values,
    class_name=train_df["Class"].values,
    category=train_df["Category"].values,
    plant_part=train_df["Plant_Part"].values,
    path=train_df["path"].values
)


# ------------------------------------------------------------
# Validation NPZ
# ------------------------------------------------------------

np.savez_compressed(
    validation_npz_path,
    embeddings=validation_embeddings,
    class_id=val_df["class_id"].values,
    class_name=val_df["Class"].values,
    category=val_df["Category"].values,
    plant_part=val_df["Plant_Part"].values,
    path=val_df["path"].values
)


# ------------------------------------------------------------
# Test NPZ
# ------------------------------------------------------------

np.savez_compressed(
    test_npz_path,
    embeddings=test_embeddings,
    class_id=test_df["class_id"].values,
    class_name=test_df["Class"].values,
    category=test_df["Category"].values,
    plant_part=test_df["Plant_Part"].values,
    path=test_df["path"].values
)


print("\nSaved NPZ embeddings:")

print(
    train_npz_path
)

print(
    validation_npz_path
)

print(
    test_npz_path
)


# ============================================================
# 4.7 SAVE TRAINING EMBEDDINGS AS CSV
# ============================================================

embedding_columns = [
    f"embedding_{i}"
    for i in range(EMBEDDING_DIM)
]


train_embedding_df = pd.DataFrame(
    train_embeddings,
    columns=embedding_columns
)


# ------------------------------------------------------------
# Add metadata columns
# ------------------------------------------------------------

train_embedding_df.insert(
    0,
    "class_id",
    train_df["class_id"].values
)

train_embedding_df.insert(
    1,
    "Class",
    train_df["Class"].values
)

train_embedding_df.insert(
    2,
    "Category",
    train_df["Category"].values
)

train_embedding_df.insert(
    3,
    "Plant_Part",
    train_df["Plant_Part"].values
)

train_embedding_df.insert(
    4,
    "path",
    train_df["path"].values
)


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

train_csv_path = os.path.join(
    OUTPUT_FOLDER,
    "train_cae_embeddings.csv"
)


train_embedding_df.to_csv(
    train_csv_path,
    index=False
)


print(
    "\nSaved training embedding CSV:",
    train_csv_path
)


# ============================================================
# 4.8 VERIFY EMBEDDING / METADATA ALIGNMENT
# ============================================================

assert len(train_embeddings) == len(train_df)

assert len(validation_embeddings) == len(val_df)

assert len(test_embeddings) == len(test_df)


assert len(train_embedding_df) == len(train_df)


print(
    "\nEmbedding / metadata alignment verified."
)


# ============================================================
# 4.9 CHECK SAVED NPZ CONTENTS
# ============================================================

print(
    "\nChecking saved training NPZ..."
)


check_npz = np.load(
    train_npz_path,
    allow_pickle=True
)


print(
    "NPZ arrays:",
    check_npz.files
)


for key in check_npz.files:

    print(
        key,
        check_npz[key].shape
    )


# ============================================================
# 4.10 CHECK SAVED CSV CONTENTS
# ============================================================

print(
    "\nTraining CSV columns:"
)

print(
    train_embedding_df.columns.tolist()
)


# ============================================================
# 4.11 FINAL SUMMARY
# ============================================================

print("\n========================================")
print("CAE EXPERIMENT COMPLETE")
print("========================================")

print( "Stage 1: CAE reconstruction pretraining")

print("Stage 2: Supervised disease classification fine-tuning")

print(
    f"Final representation: "
    f"{EMBEDDING_DIM}-D fine-tuned CAE bottleneck"
)

print( "Saved model:", best_finetuned_cae_path)

print("\nSaved embeddings:")

print(train_npz_path)

print(validation_npz_path)

print(test_npz_path)

print("\nSaved training CSV:")

print(train_csv_path)

print(
    "\nEmbeddings are ready for the same "
    "StandardScaler + clustering pipeline "
    "used for ViT and ResNet."
)
