# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Grocery Assistant is a point-and-ask visual question answering (VQA) system for
blind and low-vision users: photograph a grocery product, ask a question by voice
("is this expired?", "what flavor is this?"), get a spoken answer. It fine-tunes a
VQA model on real blind-user questions (VizWiz) and grounds answers in dedicated
product detection and expiration-date OCR rather than having the VLM guess freely.

## Status

The project is in early scaffolding. Only Phase 1 (setup) and Phase 2 (data
acquisition notebook) are done — `src/detection`, `src/ocr`, `src/voice`,
`src/vqa`, and `app/` are still empty. Build proceeds in 8 tracked phases (see
README.md "Status" section for the current list) — check there before assuming
a component exists.

## Architecture

Pipeline, in order:

```
Photo + spoken question
  -> Whisper (speech -> text)                                  [Phase 6]
  -> YOLOv8 product detection -> OCR label/expiration reading    [Phases 3-4]
  -> BLIP + LoRA VQA, grounded in detection/OCR output            [Phase 5]
  -> TTS (text -> speech)                                        [Phase 6]
  -> Spoken answer
```

Each pipeline stage is an independently developed/trained component before
integration (Phase 7) and demo app wiring (Phase 8) — don't assume cross-stage
coupling exists yet unless the relevant phase is complete.

Directory-to-phase mapping:
- `src/detection/` — YOLOv8 product detection, fine-tuned on SKU-110K (Colab)
- `src/ocr/` — OpenCV preprocessing (deskew/denoise) + EasyOCR/Tesseract for
  label and expiration-date reading, validated on ExpDate dataset
- `src/vqa/` — BLIP fine-tuned on VizWiz-VQA via LoRA (Hugging Face PEFT), Colab
  training, tracked in Weights & Biases
- `src/voice/` — Whisper STT + TTS
- `app/` — Streamlit/Gradio demo app tying the pipeline together
- `notebooks/` — Colab-run notebooks (data prep, training); `01_data_acquisition.ipynb`
  downloads/inspects all four datasets and is meant to run in Colab, not locally
- `data/raw/`, `data/processed/`, `models/` — gitignored except `.gitkeep`; large
  data/model artifacts are never committed, only referenced by version/source in
  README or notebook

## Data sources

- VizWiz-VQA — real blind-user photos/questions, core VQA fine-tuning set
- SKU-110K — 11,743 real supermarket shelf images, 1.7M bounding-box
  annotations, single-class "product" detection — primary YOLOv8 detection
  training set. Freiburg is classification-only (no bounding boxes), so it
  can't train a detector; SKU-110K supplies the boxes YOLO needs.
- Freiburg Groceries — 5,000 images, 25 grocery classes; optional supplementary
  classification signal only, not the detection training set
- ExpDate — expiration date recognition (start with Products-Real subset)
- Grozi-120 / Grocery Products — product recognition (download link needs
  reconfirming before Phase 3, per notebook TODO — original hosting has moved)

## Environment

No dedicated dev/test tooling exists yet (no test suite, linter config, or
build script). Training and data-heavy notebook work is designed to run on
Google Colab (GPU for Phases 3/5, CPU-only fine for Phase 2's data acquisition),
not locally. Install dependencies with `pip install -r requirements.txt`.

Secrets (W&B, HF Hub API keys, etc.) go in `.env`, never committed.

## Out of scope for v1

Real-time video processing, on-device/mobile deployment, and full
productionization (Docker, CI/CD, hosted API, monitoring) are deliberately
deferred.
