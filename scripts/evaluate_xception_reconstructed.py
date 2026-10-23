from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

SEED = 42
CLASSES = ["N", "D", "G", "C", "A", "H", "M"]
ROOT = Path(r"D:\gpt\OcuVision-AI")
DATA_DIR = ROOT / "data" / "preprocessed"
CHECKPOINT = ROOT / "models" / "xception_best_checkpoint.pth"
OUT_DIR = ROOT / "results" / "xception"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Sort paths for a deterministic reconstructed split.
records = []
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

for label, cls in enumerate(CLASSES):
    paths = []
    for ext in extensions:
        paths.extend((DATA_DIR / cls).glob(ext))
    for path in sorted(set(paths), key=lambda p: str(p).lower()):
        records.append({
            "image_path": str(path),
            "class": cls,
            "label": label
        })

df = pd.DataFrame(records)
print("Total images:", len(df))

train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)
test_df = test_df.reset_index(drop=True)
print("Reconstructed split sizes:", len(train_df), len(val_df), len(test_df))

class FundusDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as im:
            image = im.convert("RGB")
        return self.transform(image), int(row["label"]), row["image_path"]

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = timm.create_model("xception", pretrained=False, num_classes=len(CLASSES))
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(device).eval()

loader = DataLoader(
    FundusDataset(test_df, transform),
    batch_size=32,
    shuffle=False,
    num_workers=0
)

y_true, y_pred, rows = [], [], []
with torch.inference_mode():
    for images, labels, paths in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu().tolist()
        labels = labels.tolist()
        y_true.extend(labels)
        y_pred.extend(preds)
        rows.extend(zip(paths, labels, preds))

print("\n=== RECONSTRUCTED TEST RESULTS ===")
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Saved validation Macro F1:", checkpoint.get("best_val_macro_f1"))
print("Test images:", len(y_true))
print(f"Accuracy:        {accuracy_score(y_true, y_pred)*100:.4f}%")
print(f"Macro precision: {precision_score(y_true, y_pred, average='macro', zero_division=0)*100:.4f}%")
print(f"Macro recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0)*100:.4f}%")
print(f"Macro F1:        {f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.4f}%")
print("\nClassification report:")
print(classification_report(
    y_true, y_pred, labels=list(range(len(CLASSES))),
    target_names=CLASSES, zero_division=0, digits=4
))
print("Confusion matrix (true rows, predicted columns; class order N,D,G,C,A,H,M):")
print(confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))))

pred_df = pd.DataFrame([
    {
        "image_path": path,
        "true_class": CLASSES[true],
        "predicted_class": CLASSES[pred],
        "correct": true == pred
    }
    for path, true, pred in rows
])
csv_path = OUT_DIR / "reconstructed_test_predictions.csv"
pred_df.to_csv(csv_path, index=False)
print("\nPredictions saved to:", csv_path)
print("IMPORTANT: this is a reconstructed split, not proof of the original official test split.")
