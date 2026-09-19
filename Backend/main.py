import io
import os
import re

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    BitForImageClassification,
)
from torchvision import transforms
import pytesseract


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]

OCR_AVAILABLE = False

for tesseract_executable in TESSERACT_PATHS:
    if os.path.exists(tesseract_executable):
        pytesseract.pytesseract.tesseract_cmd = tesseract_executable
        OCR_AVAILABLE = True
        break


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="MISAFE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DEVICE / PATHS
# ============================================================

device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

TEXT_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "text_model_final",
)

IMAGE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "image_model_final",
)


# ============================================================
# LOAD TEXT MODEL
# ============================================================

print("Loading Text Model...")

if not os.path.isdir(TEXT_MODEL_PATH):
    raise FileNotFoundError(
        f"Text model folder not found: {TEXT_MODEL_PATH}"
    )

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_PATH
)

text_classifier = (
    AutoModelForSequenceClassification
    .from_pretrained(TEXT_MODEL_PATH)
    .to(device)
    .eval()
)

text_backbone = (
    AutoModel
    .from_pretrained(TEXT_MODEL_PATH)
    .to(device)
    .eval()
)


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

print("Loading Image Model...")

if not os.path.isdir(IMAGE_MODEL_PATH):
    raise FileNotFoundError(
        f"Image model folder not found: {IMAGE_MODEL_PATH}"
    )

image_classifier = (
    BitForImageClassification
    .from_pretrained(IMAGE_MODEL_PATH)
    .to(device)
    .eval()
)

image_backbone = image_classifier.bit


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ============================================================
# MULTIMODAL CLASSIFIER
#
# 768 text features + 2048 image features = 2816
#
# IMPORTANT:
# Class 0 = Non-Misogynous
# Class 1 = Misogynous
# This MUST match the class order used during training.
# ============================================================

class MultimodalClassifier(nn.Module):

    def __init__(
        self,
        input_dim=2816,
        hidden_dim=512,
        num_classes=2,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================================
# LOAD MULTIMODAL MODEL
# ============================================================

print("Loading Multimodal Model...")

model_weights_path = None

for root, dirs, files in os.walk(MODELS_DIR):
    if "multimodal_model.pt" in files:
        model_weights_path = os.path.join(
            root,
            "multimodal_model.pt",
        )
        break

if not model_weights_path:
    raise FileNotFoundError(
        f"multimodal_model.pt not found inside: {MODELS_DIR}"
    )

multimodal_model = MultimodalClassifier().to(device)

state_dict = torch.load(
    model_weights_path,
    map_location=device,
)

multimodal_model.load_state_dict(
    state_dict
)

multimodal_model.eval()


# ============================================================
# OCR CLEANING
# ============================================================

def clean_ocr_text(raw_text: str) -> str:

    if not raw_text:
        return ""

    cleaned = re.sub(
        r"[^a-zA-Z0-9\s.,?!]",
        " ",
        raw_text,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    return cleaned


# ============================================================
# IMAGE VALIDATION
# ============================================================

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def home():

    return {
        "status": "MISAFE API Running",
        "ocr_available": OCR_AVAILABLE,
        "device": str(device),
    }


# ============================================================
# MULTIMODAL PREDICTION
# ============================================================

@app.post("/predict/multimodal")
async def predict_multimodal(
    text: str = Form(None),
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image format. "
                "Use JPG, JPEG, PNG or WEBP."
            ),
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    img_bytes = await file.read()

    if not img_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(img_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB.",
        )

    try:
        image = Image.open(
            io.BytesIO(img_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read image: {exc}",
        )


    # --------------------------------------------------------
    # TEXT / OCR
    # --------------------------------------------------------

    extracted_text = text

    if (
        not extracted_text
        or extracted_text.strip() == ""
    ):

        if OCR_AVAILABLE:

            try:

                raw_ocr = pytesseract.image_to_string(
                    image
                )

                extracted_text = clean_ocr_text(
                    raw_ocr
                )

            except Exception as exc:

                print(
                    f"[OCR WARNING] {exc}"
                )

                extracted_text = ""


    # Keep the same fallback used by the original system.
    if (
        not extracted_text
        or len(extracted_text.strip()) < 3
    ):

        extracted_text = "neutral image context"


    print(
        f"\n[CLEANED PROCESSED TEXT]: "
        f"'{extracted_text}'"
    )


    # --------------------------------------------------------
    # TOKENIZE TEXT
    # --------------------------------------------------------

    inputs = tokenizer(
        extracted_text,
        return_tensors="pt",
        truncation=True,
        max_length=160,
        padding=False,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # IMAGE TRANSFORM
    # --------------------------------------------------------

    img_tensor = (
        img_transforms(image)
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------------------
    # MULTIMODAL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        # Text feature: 768 dimensions
        t_output = text_backbone(
            **inputs
        )

        t_feat = (
            t_output
            .last_hidden_state[:, 0, :]
        )

        # Image feature: expected 2048 dimensions
        i_output = image_backbone(
            img_tensor
        )

        i_feat = i_output.pooler_output

        # Handle possible [B,C,1,1] output safely.
        if i_feat.ndim > 2:
            i_feat = i_feat.flatten(
                start_dim=1
            )

        if t_feat.ndim > 2:
            t_feat = t_feat.flatten(
                start_dim=1
            )

        # ----------------------------------------------------
        # Safety check for trained multimodal architecture
        # ----------------------------------------------------

        if t_feat.shape[1] + i_feat.shape[1] != 2816:
            raise RuntimeError(
                "Feature dimension mismatch. "
                f"Text={t_feat.shape[1]}, "
                f"Image={i_feat.shape[1]}, "
                f"Expected total=2816."
            )

        fusion = torch.cat(
            (t_feat, i_feat),
            dim=1,
        )

        logits = multimodal_model(
            fusion
        )

        probs = torch.softmax(
            logits,
            dim=-1,
        )[0]


    # --------------------------------------------------------
    # CLASS PROBABILITIES
    #
    # Ground-truth class mapping:
    # 0 = Non-Misogynous
    # 1 = Misogynous
    # --------------------------------------------------------

    prob_c0 = float(
        probs[0].item()
    ) * 100

    prob_c1 = float(
        probs[1].item()
    ) * 100


    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    if probs[1] > probs[0]:

        final_verdict = "Misogynous"
        final_confidence = round(
            prob_c1,
            2,
        )

    else:

        final_verdict = "Non-Misogynous"
        final_confidence = round(
            prob_c0,
            2,
        )


    # --------------------------------------------------------
    # LOGS
    # --------------------------------------------------------

    print(
        "[SCORES] "
        f"Class 0 (Non-Misogynous): "
        f"{prob_c0:.2f}% | "
        f"Class 1 (Misogynous): "
        f"{prob_c1:.2f}%"
    )

    print(
        f"[FINAL VERDICT] -> "
        f"{final_verdict} "
        f"({final_confidence}%)\n"
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "prediction": final_verdict,
        "confidence": final_confidence,

        "extracted_text": extracted_text,

        "class_probs": {
            "Non-Misogynous": round(
                prob_c0,
                2,
            ),
            "Misogynous": round(
                prob_c1,
                2,
            ),
        },
    }
