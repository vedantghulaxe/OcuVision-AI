from pathlib import Path
import random
import json
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(r"D:\gpt\OcuVision-AI")
DATA = ROOT / "data" / "preprocessed_clean"
OUT = ROOT / "results" / "xception_clean"
MODEL_DIR = ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 299
BATCH_SIZE = 16
EPOCHS = 3
CLASSES = ["N", "D", "G", "C", "A", "H", "M"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Collect files in a fixed order and retain the specified class order.
records = []
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for label_id, label in enumerate(CLASSES):
    folder = DATA / label
    if not folder.exists():
        raise FileNotFoundError(f"Missing class folder: {folder}")
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            records.append({"path": str(path), "label": label_id})

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError(f"No images found in {DATA}")

print("\nDataset counts:")
for i, label in enumerate(CLASSES):
    print(f"{label}: {(df.label == i).sum()}")
print("Total:", len(df))

# Stratified 80/10/10 split.
train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print(f"{name}: {len(part)}")

# Save the exact split membership for reproducibility.
split_df = pd.concat([
    train_df.assign(split="train"),
    val_df.assign(split="validation"),
    test_df.assign(split="test")
])
split_df["class_name"] = split_df["label"].map(dict(enumerate(CLASSES)))
split_df.to_csv(OUT / "split_manifest.csv", index=False)

# Check no exact file path appears in multiple splits.
assert set(train_df.path).isdisjoint(val_df.path)
assert set(train_df.path).isdisjoint(test_df.path)
assert set(val_df.path).isdisjoint(test_df.path)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class FundusDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, int(row.label)

train_loader = DataLoader(
    FundusDataset(train_df, train_transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    pin_memory=(device.type == "cuda")
)
val_loader = DataLoader(
    FundusDataset(val_df, eval_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=(device.type == "cuda")
)
test_loader = DataLoader(
    FundusDataset(test_df, eval_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    pin_memory=(device.type == "cuda")
)

model = timm.create_model("xception", pretrained=True, num_classes=len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

def run_epoch(loader, training):
    model.train(training)
    losses, y_true, y_pred = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        predictions = outputs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=list(range(len(CLASSES))),
            average="macro", zero_division=0
        ),
        "y_true": y_true,
        "y_pred": y_pred
    }

best_f1 = -1.0
best_path = MODEL_DIR / "xception_clean_best_checkpoint.pth"
history = []

for epoch in range(1, EPOCHS + 1):
    train_metrics = run_epoch(train_loader, True)
    val_metrics = run_epoch(val_loader, False)

    history.append({
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"]
    })

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train loss {train_metrics['loss']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} | "
        f"val accuracy {val_metrics['accuracy']:.4%} | "
        f"val macro F1 {val_metrics['macro_f1']:.4%}"
    )

    if val_metrics["macro_f1"] > best_f1:
        best_f1 = val_metrics["macro_f1"]
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "img_size": IMG_SIZE,
            "epoch": epoch,
            "best_val_macro_f1": best_f1,
            "seed": SEED
        }, best_path)

pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)

# Load the best validation checkpoint and evaluate the test set once.
checkpoint = torch.load(best_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
test_metrics = run_epoch(test_loader, False)

summary = {
    "test_accuracy": float(test_metrics["accuracy"]),
    "test_macro_precision": float(precision_score(
        test_metrics["y_true"], test_metrics["y_pred"],
        average="macro", zero_division=0
    )),
    "test_macro_recall": float(recall_score(
        test_metrics["y_true"], test_metrics["y_pred"],
        average="macro", zero_division=0
    )),
    "test_macro_f1": float(test_metrics["macro_f1"]),
    "test_weighted_f1": float(f1_score(
        test_metrics["y_true"], test_metrics["y_pred"],
        average="weighted", zero_division=0
    )),
    "test_images": len(test_df),
    "best_epoch": int(checkpoint["epoch"]),
    "best_validation_macro_f1": float(checkpoint["best_val_macro_f1"]),
    "classes": CLASSES,
    "seed": SEED
}

(OUT / "test_metrics.json").write_text(json.dumps(summary, indent=2))
report = classification_report(
    test_metrics["y_true"], test_metrics["y_pred"],
    labels=list(range(len(CLASSES))), target_names=CLASSES,
    output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv(OUT / "test_classification_report.csv")
pd.DataFrame(
    confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"],
                     labels=list(range(len(CLASSES)))),
    index=CLASSES, columns=CLASSES
).to_csv(OUT / "test_confusion_matrix.csv")

pred_df = test_df.reset_index(drop=True).copy()
pred_df["true_class"] = [CLASSES[i] for i in test_metrics["y_true"]]
pred_df["predicted_class"] = [CLASSES[i] for i in test_metrics["y_pred"]]
pred_df["correct"] = pred_df["true_class"] == pred_df["predicted_class"]
pred_df.to_csv(OUT / "test_predictions.csv", index=False)

print("\nFINAL TEST METRICS")
print(json.dumps(summary, indent=2))
print("\nSaved best checkpoint:", best_path)
print("Saved results folder:", OUT)

