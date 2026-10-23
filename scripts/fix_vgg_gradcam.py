import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

ROOT = r"D:\gpt\OcuVision-AI"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["N", "D", "G", "C", "A", "H", "M"]

XAI_CSV = os.path.join(
    ROOT,
    "results",
    "cross_model_xai",
    "common_xai_test_images.csv"
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "results",
    "cross_model_xai",
    "corrected_vgg_gradcam"
)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

VGG16_CHECKPOINT = os.path.join(
    ROOT, "models", "vgg16_best_checkpoint.pth"
)

VGG19_CHECKPOINT = os.path.join(
    ROOT, "models", "vgg19_best_checkpoint.pth"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_index):

        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[0, class_index]

        score.backward()

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# LOAD VGG16
# ============================================================

def load_vgg16():

    print("\nLoading VGG16...")

    model = models.vgg16(weights=None)

    model.classifier[6] = torch.nn.Linear(
        model.classifier[6].in_features,
        len(CLASSES)
    )

    checkpoint = torch.load(
        VGG16_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    # IMPORTANT:
    # Correct final convolutional layer
    target_layer = model.features[28]

    print("VGG16 target layer:")
    print(target_layer)

    return model, target_layer


# ============================================================
# LOAD VGG19
# ============================================================

def load_vgg19():

    print("\nLoading VGG19...")

    model = models.vgg19(weights=None)

    model.classifier[6] = torch.nn.Linear(
        model.classifier[6].in_features,
        len(CLASSES)
    )

    checkpoint = torch.load(
        VGG19_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    # IMPORTANT:
    # Correct final convolutional layer
    target_layer = model.features[34]

    print("VGG19 target layer:")
    print(target_layer)

    return model, target_layer


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def find_image(filename):

    possible_paths = [
        os.path.join(ROOT, "data", "preprocessed", filename),
        os.path.join(ROOT, "data", filename),
    ]

    for path in possible_paths:

        if os.path.exists(path):
            return path

    # Search recursively as fallback
    for root, dirs, files in os.walk(
        os.path.join(ROOT, "data")
    ):

        if filename in files:
            return os.path.join(root, filename)

    return None


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(image, cam):

    height, width = image.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        0.55 * image +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return overlay


# ============================================================
# PROCESS MODEL
# ============================================================

def process_model(model_name, model, target_layer):

    print("\n" + "=" * 70)
    print(f"PROCESSING {model_name}")
    print("=" * 70)

    output_dir = os.path.join(
        OUTPUT_ROOT,
        model_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    df = pd.read_csv(XAI_CSV)

    results = []

    for index, row in df.iterrows():

        filename = row["filename"]

        true_class = row["true_class"]

        image_path = find_image(filename)

        if image_path is None:

            print(
                f"[WARNING] Image not found: {filename}"
            )

            continue

        print(
            f"[{index + 1}/{len(df)}] {filename}"
        )

        original = Image.open(
            image_path
        ).convert("RGB")

        original_np = np.array(
            original
        )

        input_tensor = transform(
            original
        ).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_index = (
                probabilities.argmax(dim=1).item()
            )

            confidence = (
                probabilities[0, predicted_index]
                .item()
            )

        predicted_class = CLASSES[
            predicted_index
        ]

        # Grad-CAM
        cam = gradcam.generate(
            input_tensor,
            predicted_index
        )

        overlay = create_overlay(
            original_np,
            cam
        )

        # Save comparison
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        axes[0].imshow(
            original_np
        )

        axes[0].set_title(
            f"Original\nTrue: {true_class}"
        )

        axes[1].imshow(
            cam,
            cmap="jet"
        )

        axes[1].set_title(
            f"Grad-CAM\nPred: {predicted_class}"
        )

        axes[2].imshow(
            overlay
        )

        axes[2].set_title(
            f"{model_name}\n"
            f"{predicted_class} "
            f"({confidence * 100:.2f}%)"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"{index + 1:02d}_{os.path.splitext(filename)[0]}_gradcam.png"
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        results.append({
            "xai_id": f"XAI_{index + 1:02d}",
            "filename": filename,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "correct": predicted_class == true_class,
            "gradcam_path": save_path
        })

    gradcam.close()

    results_df = pd.DataFrame(
        results
    )

    csv_path = os.path.join(
        output_dir,
        f"{model_name}_corrected_gradcam_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nSaved results: {csv_path}"
    )

    print(
        f"Correct: "
        f"{results_df['correct'].sum()}/"
        f"{len(results_df)}"
    )

    print(
        f"Accuracy: "
        f"{results_df['correct'].mean() * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CORRECTED VGG16 / VGG19 GRAD-CAM")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

# -------------------------
# VGG16
# -------------------------

vgg16, vgg16_target = load_vgg16()

process_model(
    "VGG16",
    vgg16,
    vgg16_target
)

# Free GPU memory
del vgg16

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# -------------------------
# VGG19
# -------------------------

vgg19, vgg19_target = load_vgg19()

process_model(
    "VGG19",
    vgg19,
    vgg19_target
)

del vgg19

if torch.cuda.is_available():
    torch.cuda.empty_cache()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Output directory:\n{OUTPUT_ROOT}"
)