from __future__ import annotations

import io
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import models, transforms

try:
    import timm
except ImportError:
    timm = None

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    PageBreak,
)

# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OcuVision-AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "OcuVision-AI"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# IMPORTANT:
# These must exactly match the classes used when the models
# were trained, including their order.
CLASSES = ["N", "D", "G", "C", "A", "H", "M"]

CLASS_NAMES = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Myopia",
}

MODEL_NAMES = [
    "Xception",
    "VGG16",
    "VGG19",
    "ResNet-50",
    "ViT-B/16",
    "InceptionV3",
]

CHECKPOINT_CANDIDATES = {
    "Xception": [
        "xception_fold1_improved_best_checkpoint.pth",
        "xception_best_checkpoint.pth",
        "xception_clean_best_checkpoint.pth",
        "xception_best.pth",
        "xception.pth",
    ],
    "VGG16": [
        "vgg16_best_checkpoint.pth",
        "vgg16_best.pth",
        "vgg16.pth",
    ],
    "VGG19": [
        "vgg19_best_checkpoint.pth",
        "vgg19_best.pth",
        "vgg19.pth",
    ],
    "ResNet-50": [
        "resnet50_best_checkpoint.pth",
        "resnet50_best.pth",
        "resnet50.pth",
    ],
    "ViT-B/16": [
        "vit_b16_best_checkpoint.pth",
        "vit_b16_best.pth",
        "vit_b16.pth",
    ],
    "InceptionV3": [
        "inception_v3_best_checkpoint.pth",
        "inceptionv3_best_checkpoint.pth",
        "inceptionv3_best.pth",
        "inceptionv3.pth",
        "inception_v3.pth",
    ],
}

# These are historical metrics supplied for the project.
# They are not recalculated from a newly uploaded image.
# Replace them with metrics from the final test set if needed.
MODEL_PERFORMANCE = {
    "Xception": {
        "accuracy": 83.44,
        "precision": 88.00,
        "recall": 83.83,
        "macro_f1": 84.98,
        "weighted_f1": 83.35,
    },
    "VGG16": {
        "accuracy": 84.880383,
        "precision": 87.191636,
        "recall": 83.177968,
        "macro_f1": 84.880028,
        "weighted_f1": 84.861382,
    },
    "VGG19": {
        "accuracy": 84.401914,
        "precision": 86.816977,
        "recall": 83.440981,
        "macro_f1": 84.800934,
        "weighted_f1": 84.331542,
    },
    "ResNet-50": {
        "accuracy": 81.24,
        "precision": 81.07,
        "recall": 85.51,
        "macro_f1": 82.58,
        "weighted_f1": 81.30,
    },
    "ViT-B/16": {
        "accuracy": 77.70,
        "precision": 83.08,
        "recall": 79.58,
        "macro_f1": 80.61,
        "weighted_f1": 77.84,
    },
    "InceptionV3": {
        "accuracy": 54.928230,
        "precision": 57.711649,
        "recall": 54.698703,
        "macro_f1": 50.932866,
        "weighted_f1": 52.969043,
    },
}

XCEPTION_HISTORY_FILE = RESULTS_DIR / "xception" / "training_history.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MAX_UPLOAD_MB = 20

# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Build a model architecture with the configured output size."""

    if model_name == "Xception":
        if timm is None:
            raise ImportError(
                "Xception requires timm. Install it with: pip install timm"
            )
        return timm.create_model(
            "xception",
            pretrained=False,
            num_classes=num_classes,
        )

    if model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )
        return model

    if model_name == "VGG19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )
        return model

    if model_name == "ResNet-50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "ViT-B/16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, num_classes
        )
        return model

    if model_name == "InceptionV3":
        # Keep aux_logits enabled to support checkpoints trained
        # with the standard torchvision InceptionV3 architecture.
        model = models.inception_v3(
            weights=None,
            aux_logits=True,
            init_weights=False,
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(
                model.AuxLogits.fc.in_features, num_classes
            )

        return model

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# CHECKPOINT DISCOVERY AND LOADING
# ============================================================

def find_checkpoint(model_name: str) -> Path | None:
    """Find a checkpoint in the project's models directory."""

    for filename in CHECKPOINT_CANDIDATES[model_name]:
        path = MODEL_DIR / filename
        if path.is_file():
            return path

    # Search recursively as a fallback.
    for filename in CHECKPOINT_CANDIDATES[model_name]:
        matches = list(BASE_DIR.rglob(filename))
        if matches:
            return matches[0]

    return None


def extract_state_dict(checkpoint: Any) -> dict:
    """Support raw state dictionaries and common training checkpoint formats."""

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "model_state",
        "net",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break

    if not isinstance(checkpoint, dict):
        raise ValueError("Could not extract model weights from checkpoint.")

    # Remove DataParallel prefixes when present.
    cleaned = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            continue

        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


@st.cache_resource(show_spinner=False)
def load_model_cached(
    model_name: str,
    checkpoint_path: str,
    checkpoint_mtime: float,
    num_classes: int,
):
    """Load a model once and cache it for subsequent predictions."""

    del checkpoint_mtime  # Included in the cache key to refresh changed files.

    model = build_model(model_name, num_classes)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = extract_state_dict(checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as strict_error:
        # A non-strict load can hide important architecture differences.
        # Only use it if the checkpoint has matching parameter tensors.
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)

            if not any(
                key in state_dict
                for key in model.state_dict().keys()
            ):
                raise strict_error

            if len(incompatible.missing_keys) > 20:
                raise RuntimeError(
                    "Too many checkpoint keys are missing. "
                    "The architecture may not match the training setup."
                ) from strict_error

        except Exception as load_error:
            raise RuntimeError(
                f"Could not load {model_name} checkpoint. "
                f"Check the architecture and class count. "
                f"Details: {strict_error}"
            ) from load_error

    model.to(DEVICE)
    model.eval()
    return model


def get_model(model_name: str):
    checkpoint = find_checkpoint(model_name)

    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found for {model_name}. "
            f"Expected one of: "
            f"{', '.join(CHECKPOINT_CANDIDATES[model_name])} "
            f"inside {MODEL_DIR}"
        )

    return load_model_cached(
        model_name,
        str(checkpoint),
        checkpoint.stat().st_mtime,
        len(CLASSES),
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Default 224x224 preprocessing retained for the existing models."""
    rgb = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(rgb).unsqueeze(0)
    return tensor.to(DEVICE)


def preprocess_for_model(image: Image.Image, model_name: str) -> torch.Tensor:
    """Use the improved Xception training resolution; preserve others."""
    if model_name == "Xception":
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    return preprocess_image(image)


def normalize_model_output(output: Any) -> torch.Tensor:
    """Extract logits from common torchvision model output formats."""

    if hasattr(output, "logits"):
        output = output.logits

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError("Model output is not a tensor.")

    return output


def predict_probabilities(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    """Return class probabilities for one image."""

    with torch.inference_mode():
        output = model(input_tensor)
        logits = normalize_model_output(output)
        probabilities = torch.softmax(logits, dim=1)[0]

    return probabilities.detach().cpu().numpy()


# ============================================================
# GRAD-CAM
# ============================================================

def get_gradcam_layer(model_name: str, model: nn.Module):
    """Select a convolutional layer appropriate for Grad-CAM."""

    if model_name == "Xception":
        # timm's Xception layout varies by version.
        for name in ("conv4", "act4", "conv3"):
            layer = getattr(model, name, None)
            if isinstance(layer, nn.Module):
                return layer

        modules = [
            module for module in model.modules()
            if isinstance(module, nn.Conv2d)
        ]
        return modules[-1] if modules else None

    if model_name in ("VGG16", "VGG19"):
        return model.features[-1]

    if model_name == "ResNet-50":
        return model.layer4[-1]

    if model_name == "InceptionV3":
        return model.Mixed_7c

    # A conventional convolutional Grad-CAM is not appropriate for
    # ViT's token-based representation. Do not show a fake heatmap.
    if model_name == "ViT-B/16":
        return None

    return None


def make_heatmap(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Convert activations and gradients into a normalized CAM."""

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(
        cam,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    cam = cam[0, 0].detach().float().cpu().numpy()

    minimum = float(cam.min())
    maximum = float(cam.max())
    cam = (cam - minimum) / (maximum - minimum + 1e-8)

    return cam


def generate_gradcam(
    model_name: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray | None:
    """Generate Grad-CAM using a tensor gradient hook safe for Xception."""
    target_layer = get_gradcam_layer(model_name, model)
    if target_layer is None:
        return None

    cache: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        if torch.is_tensor(output):
            cache["activation"] = output
            if output.requires_grad:
                # Tensor hook avoids the view/in-place issue from module backward hooks.
                output.register_hook(
                    lambda grad: cache.__setitem__("gradient", grad)
                )

    forward_handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = input_tensor.detach().clone().requires_grad_(True)
            output = normalize_model_output(model(x))
            output[0, target_class].backward()

        activation = cache.get("activation")
        gradient = cache.get("gradient")
        if activation is None or gradient is None:
            return None
        if activation.ndim != 4 or gradient.ndim != 4:
            return None

        return make_heatmap(
            activation, gradient, (IMAGE_SIZE, IMAGE_SIZE)
        )
    except Exception:
        return None
    finally:
        forward_handle.remove()
        model.zero_grad(set_to_none=True)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    """Create a visible heatmap overlay without requiring matplotlib."""

    base = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    cam = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

    # Simple blue-to-red heatmap.
    red = np.clip(255 * (2 * heatmap), 0, 255).astype(np.uint8)
    blue = np.clip(255 * (2 * (1 - heatmap)), 0, 255).astype(np.uint8)
    green = np.clip(255 * (1 - np.abs(2 * heatmap - 1)), 0, 255).astype(np.uint8)

    colored = np.stack([red, green, blue], axis=-1)
    heat_image = Image.fromarray(colored, mode="RGB")
    return Image.blend(base, heat_image, alpha=alpha)


# ============================================================
# INFERENCE AND ENSEMBLE
# ============================================================

def run_model(
    model_name: str,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    generate_cam: bool,
) -> dict:
    start = time.time()
    model = get_model(model_name)
    # Xception was trained at 299x299; other configured models retain 224x224.
    model_input = preprocess_for_model(original_image, model_name)
    probabilities = predict_probabilities(model, model_input)
    predicted_index = int(np.argmax(probabilities))

    cam = None
    overlay = None

    if generate_cam:
        cam = generate_gradcam(
            model_name,
            model,
            model_input,
            predicted_index,
        )

        if cam is not None:
            overlay = overlay_heatmap(original_image, cam)

    return {
        "model": model_name,
        "predicted_index": predicted_index,
        "predicted_class": CLASSES[predicted_index],
        "predicted_label": CLASS_NAMES[CLASSES[predicted_index]],
        "confidence": float(probabilities[predicted_index] * 100),
        "probabilities": probabilities,
        "gradcam": cam,
        "overlay": overlay,
        "seconds": time.time() - start,
    }


def run_all_models(
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    generate_cam: bool,
) -> tuple[list[dict], list[dict]]:
    """Run all six models and create a soft-voting ensemble."""

    results = []
    errors = []

    progress = st.progress(0)
    status = st.empty()

    for index, model_name in enumerate(MODEL_NAMES):
        status.write(f"Running {model_name} ({index + 1}/{len(MODEL_NAMES)})...")

        try:
            result = run_model(
                model_name,
                input_tensor,
                original_image,
                generate_cam,
            )
            results.append(result)

        except Exception as exc:
            errors.append(
                {
                    "model": model_name,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        progress.progress((index + 1) / len(MODEL_NAMES))

    progress.empty()
    status.empty()

    if results:
        probability_matrix = np.stack(
            [result["probabilities"] for result in results],
            axis=0,
        )
        ensemble_probabilities = probability_matrix.mean(axis=0)
        ensemble_index = int(np.argmax(ensemble_probabilities))

        agreement_count = sum(
            int(result["predicted_index"] == ensemble_index)
            for result in results
        )
        agreement_total = len(results)
        agreeing_models = [
            result for result in results
            if result["predicted_index"] == ensemble_index
        ]
        # A separate descriptive statistic: average top-class confidence only
        # among models whose top prediction matches the ensemble class.
        # This is NOT the ensemble probability and is not used to modify it.
        agreeing_mean_confidence = (
            float(np.mean([result["confidence"] for result in agreeing_models]))
            if agreeing_models else 0.0
        )
        ensemble = {
            "model": "Soft-Voting Ensemble",
            "predicted_index": ensemble_index,
            "predicted_class": CLASSES[ensemble_index],
            "predicted_label": CLASS_NAMES[CLASSES[ensemble_index]],
            # Requested reporting convention: final confidence is the mean top-class
            # confidence of models voting for the majority/ensemble class.
            # Preserve the all-model averaged probability separately for transparency.
            "confidence": agreeing_mean_confidence,
            "all_models_mean_probability": float(ensemble_probabilities[ensemble_index] * 100),
            "probabilities": ensemble_probabilities,
            "agreement_count": agreement_count,
            "agreement_total": agreement_total,
            "agreement_percent": (
                100.0 * agreement_count / agreement_total
                if agreement_total else 0.0
            ),
            "agreeing_models_mean_confidence": agreeing_mean_confidence,
            "gradcam": None,
            "overlay": None,
            "seconds": sum(item["seconds"] for item in results),
        }
        results.append(ensemble)

    return results, errors


# ============================================================
# METRIC TABLES AND CSV EXPORT
# ============================================================

def metrics_dataframe() -> pd.DataFrame:
    rows = []

    for model_name in MODEL_NAMES:
        metrics = MODEL_PERFORMANCE[model_name]
        rows.append(
            {
                "Model": model_name,
                "Accuracy (%)": metrics["accuracy"],
                "Precision (%)": metrics["precision"],
                "Recall (%)": metrics["recall"],
                "Macro F1 (%)": metrics["macro_f1"],
                "Weighted F1 (%)": metrics["weighted_f1"],
            }
        )

    return pd.DataFrame(rows)


def predictions_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []

    for result in results:
        row = {
            "Model": result["model"],
            "Predicted Code": result["predicted_class"],
            "Predicted Class": result["predicted_label"],
            "Confidence (%)": round(result["confidence"], 4),
            "Agreement Count": result.get("agreement_count", ""),
            "Models Evaluated": result.get("agreement_total", ""),
            "Agreement (%)": round(result["agreement_percent"], 2)
                if "agreement_percent" in result else "",
            "Inference Time (seconds)": round(result["seconds"], 4),
        }

        for index, code in enumerate(CLASSES):
            row[f"P({code} - {CLASS_NAMES[code]}) (%)"] = round(
                float(result["probabilities"][index] * 100), 4
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# PDF REPORT
# ============================================================

def pil_image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_pdf_report(
    image: Image.Image,
    results: list[dict],
    errors: list[dict],
    uploaded_filename: str,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#163B65"),
            fontSize=22,
            leading=27,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
        )
    )

    story = []

    story.append(Paragraph("OcuVision-AI", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Retinal Image Classification and Model Comparison Report",
            styles["Heading2"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Report Overview", styles["Heading1"]))
    story.append(
        Paragraph(
            f"<b>Image file:</b> {uploaded_filename}",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Image size:</b> {image.width} × {image.height}",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Models with successful inference:</b> "
            f"{sum(r['model'] != 'Soft-Voting Ensemble' for r in results)}",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 10))

    thumb = image.convert("RGB").copy()
    thumb.thumbnail((240, 240))
    image_buffer = io.BytesIO(pil_image_to_png_bytes(thumb))
    story.append(PDFImage(image_buffer, width=2.2 * inch, height=2.2 * inch))
    story.append(Spacer(1, 12))

    # Consensus summary makes the voting result easy to understand.
    model_only = [r for r in results if r["model"] != "Soft-Voting Ensemble"]
    ensemble_result = next(
        (r for r in results if r["model"] == "Soft-Voting Ensemble"), None
    )
    if ensemble_result:
        story.append(Paragraph("Consensus Summary", styles["Heading2"]))
        story.append(Paragraph(
            f"<b>Final predicted class:</b> {ensemble_result['predicted_label']} "
            f"({ensemble_result['predicted_class']})",
            styles["BodyText"],
        ))
        story.append(Paragraph(
            f"<b>Model agreement:</b> {ensemble_result.get('agreement_count', 0)} "
            f"of {ensemble_result.get('agreement_total', len(model_only))} models "
            f"({ensemble_result.get('agreement_percent', 0):.2f}%) selected this "
            f"class as their top prediction.",
            styles["BodyText"],
        ))
        story.append(Paragraph(
            f"<b>Reported confidence (majority-voting models only):</b> "
            f"{ensemble_result['confidence']:.2f}%. This is the arithmetic mean "
            "of the individual top-class confidence scores for models whose top "
            "prediction matches the final class.",
            styles["BodyText"],
        ))
        story.append(Paragraph(
            f"<b>Mean probability across all successful models:</b> "
            f"{ensemble_result.get('all_models_mean_probability', 0.0):.2f}%. "
            "This is the mean probability assigned to the predicted class across "
            "all successful models, including models that predicted another class. "
            "Neither score is a calibrated measure of clinical certainty.",
            styles["BodyText"],
        ))
        agree_names = [r["model"] for r in model_only
                       if r.get("predicted_index") == ensemble_result.get("predicted_index")]
        disagree_names = [r["model"] for r in model_only
                          if r.get("predicted_index") != ensemble_result.get("predicted_index")]
        story.append(Paragraph(
            "<b>Models agreeing:</b> " + (", ".join(agree_names) if agree_names else "None"),
            styles["BodyText"],
        ))
        story.append(Paragraph(
            "<b>Models with a different top prediction:</b> "
            + (", ".join(disagree_names) if disagree_names else "None"),
            styles["BodyText"],
        ))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Interpretation note: model agreement summarizes consensus within this "
            "ensemble. It does not establish clinical certainty or replace a "
            "validated diagnostic assessment.",
            styles["SmallNote"],
        ))
        story.append(Spacer(1, 10))

    story.append(Paragraph("2. Prediction Results", styles["Heading1"]))

    result_rows = [[
        "Model",
        "Predicted Class",
        "Code",
        "Probability (%)",
        "Agreement",
        "Agreement (%)",
        "Time (s)",
    ]]

    for result in results:
        result_rows.append(
            [
                result["model"],
                result["predicted_label"],
                result["predicted_class"],
                f"{result['confidence']:.2f}",
                (
                    f"{result['agreement_count']}/{result['agreement_total']}"
                    if "agreement_count" in result else "—"
                ),
                (
                    f"{result['agreement_percent']:.2f}"
                    if "agreement_percent" in result else "—"
                ),
                f"{result['seconds']:.2f}",
            ]
        )

    result_table = Table(
        result_rows,
        repeatRows=1,
        colWidths=[1.05 * inch, 1.15 * inch, 0.4 * inch, 0.8 * inch, 0.62 * inch, 0.7 * inch, 0.55 * inch],
    )
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163B65")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F2F6FA"),
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Per-Class Probability Distribution", styles["Heading1"]))
    story.append(Paragraph(
        "Each cell is the probability assigned to that class by the individual "
        "model. The Soft-Voting Ensemble row is the arithmetic mean of the "
        "successful models' class-probability vectors. Probabilities are model "
        "outputs and should not be interpreted as calibrated clinical risk.",
        styles["SmallNote"],
    ))
    story.append(Spacer(1, 6))

    probability_rows = [["Model"] + [
        f"{code}\n{CLASS_NAMES[code]}" for code in CLASSES
    ]]

    for result in results:
        probability_rows.append(
            [result["model"]]
            + [
                f"{float(p) * 100:.1f}%"
                for p in result["probabilities"]
            ]
        )

    probability_table = Table(
        probability_rows,
        repeatRows=1,
        colWidths=[1.45 * inch] + [0.75 * inch] * len(CLASSES),
    )
    probability_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163B65")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(probability_table)
    story.append(Spacer(1, 12))

    story.append(PageBreak())
    story.append(Paragraph("4. Historical Model Performance", styles["Heading1"]))
    story.append(
        Paragraph(
            "The following values are the previously supplied validation/test "
            "metrics configured in this application. They are not measured "
            "from the single uploaded image and are not recalculated by "
            "running inference.",
            styles["SmallNote"],
        )
    )
    story.append(Spacer(1, 8))

    metric_rows = [[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Macro F1",
        "Weighted F1",
    ]]

    for model_name in MODEL_NAMES:
        m = MODEL_PERFORMANCE[model_name]
        metric_rows.append(
            [
                model_name,
                f"{m['accuracy']:.2f}%",
                f"{m['precision']:.2f}%",
                f"{m['recall']:.2f}%",
                f"{m['macro_f1']:.2f}%",
                f"{m['weighted_f1']:.2f}%",
            ]
        )

    metric_table = Table(
        metric_rows,
        repeatRows=1,
        colWidths=[1.3 * inch] + [0.95 * inch] * 5,
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163B65")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F2F6FA"),
                ]),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. Grad-CAM Visualizations", styles["Heading1"]))

    cam_results = [
        result for result in results
        if result["model"] != "Soft-Voting Ensemble"
        and result.get("overlay") is not None
    ]

    if cam_results:
        for result in cam_results:
            story.append(
                Paragraph(
                    f"{result['model']} — {result['predicted_label']}",
                    styles["Heading3"],
                )
            )
            cam_bytes = io.BytesIO(
                pil_image_to_png_bytes(result["overlay"])
            )
            story.append(
                PDFImage(cam_bytes, width=3.0 * inch, height=3.0 * inch)
            )
            story.append(Spacer(1, 8))
    else:
        story.append(
            Paragraph(
                "No Grad-CAM images were available for this report.",
                styles["BodyText"],
            )
        )

    if errors:
        story.append(PageBreak())
        story.append(Paragraph("6. Model Loading or Inference Errors", styles["Heading1"]))
        for error in errors:
            story.append(
                Paragraph(
                    f"<b>{error['model']}</b>: "
                    f"{error['error']}",
                    styles["SmallNote"],
                )
            )
            story.append(Spacer(1, 5))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Important: This software is a research and educational aid. "
            "Its output is not a medical diagnosis. Images and predictions "
            "should be reviewed by a qualified eye-care professional.",
            styles["SmallNote"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


# ============================================================
# TRAINING HISTORY

# ============================================================
# TRAINING HISTORY AND STREAMLIT PAGES
# ============================================================

def find_training_history() -> Path | None:
    candidates = [
        RESULTS_DIR / "xception_fold1_improved" / "training_history.csv",
        RESULTS_DIR / "xception_fold1_improved" / "training_curves.csv",
        XCEPTION_HISTORY_FILE,
        RESULTS_DIR / "xception_training_history.csv",
        RESULTS_DIR / "training_history.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    possible = list(RESULTS_DIR.rglob("*training*history*.csv")) if RESULTS_DIR.exists() else []
    return possible[0] if possible else None


def render_training_history():
    st.subheader("Training History")
    history_path = find_training_history()
    if history_path is None:
        st.info("No training-history CSV found under results/.")
        return
    st.caption(f"Loaded from: {history_path}")
    try:
        history = pd.read_csv(history_path)
    except Exception as exc:
        st.error(f"Could not read training history: {exc}")
        return
    st.dataframe(history, use_container_width=True)
    st.download_button(
        "Download Training History CSV",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="training_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    numeric = history.select_dtypes(include=[np.number])
    for term, heading in [
        ("loss", "Loss History"),
        ("accuracy", "Accuracy History"),
        ("f1", "F1 History"),
    ]:
        columns = [c for c in numeric.columns if term in c.lower()
                   or (term == "accuracy" and c.lower().endswith("_acc"))]
        if columns:
            st.markdown(f"#### {heading}")
            st.line_chart(numeric[columns])


with st.sidebar:
    st.title("👁️ OcuVision-AI")
    st.caption("Multi-model retinal image classification")
    page = st.radio(
        "Navigation",
        ["Overview", "Run All 6 Models", "Model Comparison",
         "Training History", "Checkpoint Status", "About"],
    )
    st.divider()
    st.write(f"**Device:** `{DEVICE}`")
    st.write(f"**Class count:** {len(CLASSES)}")
    st.write(f"**Model directory:** `{MODEL_DIR}`")


if page == "Overview":
    st.title("OcuVision-AI")
    st.write(
        "A research dashboard for retinal image classification using six "
        "deep-learning architectures and a soft-voting ensemble."
    )
    st.warning(
        "Research use only. Predictions are not a medical diagnosis and "
        "must not replace professional clinical assessment."
    )
    st.subheader("Supported Models")
    cols = st.columns(3)
    for i, name in enumerate(MODEL_NAMES):
        with cols[i % 3]:
            checkpoint = find_checkpoint(name)
            if checkpoint:
                st.success(f"**{name}**\\n\\nCheckpoint found")
                st.caption(str(checkpoint))
            else:
                st.error(f"**{name}**\\n\\nCheckpoint missing")
    st.subheader("Historical Performance")
    st.dataframe(metrics_dataframe().round(2), use_container_width=True, hide_index=True)
    st.caption(
        "Historical metrics are configured project values. They are not "
        "measured from an uploaded image and should be verified against the "
        "final evaluation split."
    )
    st.subheader("Seven Classes")
    class_cols = st.columns(4)
    for i, code in enumerate(CLASSES):
        with class_cols[i % 4]:
            st.metric(code, CLASS_NAMES[code])


elif page == "Checkpoint Status":
    st.title("Checkpoint Status")
    rows = []
    for name in MODEL_NAMES:
        checkpoint = find_checkpoint(name)
        rows.append({
            "Model": name,
            "Status": "Available" if checkpoint else "Missing",
            "Checkpoint": str(checkpoint) if checkpoint else "Not found",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(
        "Checkpoint architecture, class count, class order, and preprocessing "
        "must match the training setup. Xception uses the improved checkpoint "
        "when it is present in models/."
    )


elif page == "Model Comparison":
    st.title("Model Comparison")
    metrics = metrics_dataframe()
    metric = st.selectbox(
        "Choose a metric",
        ["Accuracy (%)", "Precision (%)", "Recall (%)", "Macro F1 (%)", "Weighted F1 (%)"],
    )
    st.bar_chart(metrics.set_index("Model")[[metric]])
    st.dataframe(
        metrics.sort_values(metric, ascending=False).round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Model Metrics CSV",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_metrics.csv",
        mime="text/csv",
    )
    st.caption("These are configured historical metrics, not live single-image measurements.")


elif page == "Training History":
    st.title("Training History")
    render_training_history()


elif page == "Run All 6 Models":
    st.title("Run All 6 Models")
    st.write(
        "Upload a retinal fundus image. Available checkpoints are evaluated, "
        "then their class probabilities are averaged for a soft-voting ensemble."
    )
    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        help=f"Maximum upload size: {MAX_UPLOAD_MB} MB",
    )
    generate_cam = st.checkbox(
        "Generate Grad-CAM explanations for supported CNN models", value=True
    )

    if uploaded_file is not None:
        if len(uploaded_file.getvalue()) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit.")
            st.stop()
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as exc:
            st.error(f"Could not open uploaded image: {exc}")
            st.stop()

        col_image, col_info = st.columns([1, 2])
        with col_image:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with col_info:
            st.markdown("### Image Details")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Dimensions:** {image.width} × {image.height}")
            st.write(f"**Device:** {DEVICE}")
            st.write(f"**Configured models:** {len(MODEL_NAMES)}")

        if st.button("Run Predictions", type="primary", use_container_width=True):
            try:
                input_tensor = preprocess_image(image)
                with st.spinner("Running available models..."):
                    results, errors = run_all_models(input_tensor, image, generate_cam)
                st.session_state["ov_results"] = results
                st.session_state["ov_errors"] = errors
                st.session_state["ov_image"] = image
                st.session_state["ov_filename"] = uploaded_file.name
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.code(traceback.format_exc())

        results = st.session_state.get("ov_results")
        errors = st.session_state.get("ov_errors", [])
        saved_image = st.session_state.get("ov_image")
        saved_filename = st.session_state.get("ov_filename", "retinal_image")

        if results:
            st.divider()
            st.subheader("Prediction Summary")
            ensemble_items = [r for r in results if r["model"] == "Soft-Voting Ensemble"]
            individual_items = [r for r in results if r["model"] != "Soft-Voting Ensemble"]

            if ensemble_items:
                ensemble = ensemble_items[0]
                st.success(
                    f"Final predicted class: {ensemble['predicted_class']} — "
                    f"{ensemble['predicted_label']} | "
                    f"{ensemble['confidence']:.2f}% averaged probability"
                )
                st.markdown("#### Final Prediction Confidence Report")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Final predicted class",
                    f"{ensemble['predicted_class']} — {ensemble['predicted_label']}",
                )
                c2.metric(
                    "Majority-model mean confidence",
                    f"{ensemble['confidence']:.2f}%",
                    help="Arithmetic mean of top-class confidence scores from only the models whose predicted class matches the majority/ensemble class.",
                )
                c3.metric(
                    "Model consensus",
                    f"{ensemble['agreement_count']}/{ensemble['agreement_total']} "
                    f"({ensemble['agreement_percent']:.1f}%)",
                    help="Number and percentage of successful models whose top prediction matches the ensemble.",
                )
                c4.metric(
                    "All-model mean probability",
                    f"{ensemble.get('all_models_mean_probability', 0.0):.2f}%",
                    help="Mean probability assigned to the final class across all successful models, including dissenting models.",
                )
                st.caption(
                    "Reported confidence is the mean top-class confidence among the "
                    "majority-voting models. The all-model mean probability is shown "
                    "separately so dissenting models remain visible in the report."
                )
                st.success(
                    f"Consensus result: {ensemble['agreement_count']} of "
                    f"{ensemble['agreement_total']} models predicted "
                    f"**{ensemble['predicted_label']}**. This is a model-vote "
                    f"summary; the reported confidence is the average confidence "
                    f"of the models voting for that class: {ensemble['confidence']:.2f}%."
                )
                st.caption(
                    "Probability and agreement measure different things. Agreement "
                    "does not increase or replace probability, and neither is a "
                    "measure of clinical certainty. This tool is a research aid, "
                    "not a diagnosis."
                )

            result_rows = []
            for item in results:
                result_rows.append({
                    "Model": item["model"],
                    "Prediction": item["predicted_label"],
                    "Class Code": item["predicted_class"],
                    "Confidence (%)": round(item["confidence"], 2),
                    "Agreement": (
                        f"{item['agreement_count']}/{item['agreement_total']}"
                        if "agreement_count" in item else "—"
                    ),
                    "Agreement (%)": (
                        round(item["agreement_percent"], 2)
                        if "agreement_percent" in item else "—"
                    ),
                    "Time (s)": round(item["seconds"], 2),
                })
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)

            st.subheader("Class Probability Distribution")
            st.caption(
                "Each ensemble class probability is the mean probability from "
                "the models that completed inference successfully. Probabilities "
                "are not calibrated clinical certainty."
            )
            selected_name = st.selectbox("Choose a model", [r["model"] for r in results])
            selected_result = next(r for r in results if r["model"] == selected_name)
            prob_df = pd.DataFrame({
                "Class": [f"{code} — {CLASS_NAMES[code]}" for code in CLASSES],
                "Probability (%)": [float(p) * 100 for p in selected_result["probabilities"]],
            }).set_index("Class")
            st.bar_chart(prob_df)
            st.dataframe(prob_df.round(3), use_container_width=True)

            st.subheader("Grad-CAM Explainability")
            cam_results = [r for r in individual_items if r.get("overlay") is not None]
            if cam_results:
                cam_cols = st.columns(2)
                for i, item in enumerate(cam_results):
                    with cam_cols[i % 2]:
                        st.markdown(f"**{item['model']}**")
                        st.image(
                            item["overlay"],
                            caption=(
                                f"Prediction: {item['predicted_label']}. "
                                "Highlighted regions show model activation, "
                                "not a confirmed disease location."
                            ),
                            use_container_width=True,
                        )
            else:
                st.info(
                    "No Grad-CAM maps were generated. ViT-B/16 is transformer-based "
                    "and is not handled by this CNN Grad-CAM implementation; "
                    "other maps may be unavailable for unsupported layers."
                )

            st.subheader("Export Results")
            csv_data = predictions_dataframe(results).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions and Confidence Report CSV",
                data=csv_data,
                file_name="ocuvision_predictions_confidence_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if saved_image is not None:
                try:
                    pdf_bytes = make_pdf_report(saved_image, results, errors, saved_filename)
                    st.download_button(
                        "Download Detailed PDF Report",
                        data=pdf_bytes,
                        file_name="ocuvision_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Could not generate PDF report: {exc}")

            if errors:
                st.subheader("Models with Errors")
                for error in errors:
                    with st.expander(f"{error['model']}: {error['error']}"):
                        st.code(error["traceback"])

            st.caption(
                "Soft voting averages probabilities from successful models only. "
                "The ensemble can use fewer than six models if a checkpoint is "
                "missing or cannot be loaded."
            )
    else:
        st.info("Upload an image to enable prediction.")


elif page == "About":
    st.title("About OcuVision-AI")
    st.markdown(
        """
        OcuVision-AI is a research-oriented retinal image classification dashboard.

        **Models:** Xception, VGG16, VGG19, ResNet-50, ViT-B/16, and InceptionV3.

        **Features:** checkpoint discovery, individual predictions, probability-based
        soft-voting ensemble, model agreement report, Grad-CAM for supported CNNs,
        historical model comparison, training-history display, CSV export, and PDF report.

        **Classes:** N — Normal; D — Diabetes; G — Glaucoma; C — Cataract;
        A — Age-related Macular Degeneration; H — Hypertension; M — Myopia.
        """
    )
    st.warning(
        "This tool is for research and educational use. It is not a substitute "
        "for an ophthalmologist, clinical examination, or validated diagnostic system."
    )
    st.write(f"**Project root:** `{BASE_DIR}`")
    st.write(f"**Model directory:** `{MODEL_DIR}`")
    st.write(f"**Results directory:** `{RESULTS_DIR}`")
