# Grocery Assistant — Visual Question Answering for Blind & Low-Vision Users

A point-and-ask assistant: photograph a grocery product, ask a question by voice
("is this expired?", "what flavor is this?", "what's the price?"), and get a
spoken answer. Modeled on the real precedent of Be My Eyes / Be My AI's retail
partnerships (e.g. Tesco), scoped down to a buildable v1.

## Why this exists

Vision-language models can now answer free-form questions about images, but
general-purpose assistants aren't tuned for the specific, high-stakes questions
blind and low-vision shoppers actually ask (is this expired, what size is this,
is this the right item). This project fine-tunes a VQA model on real
blind-user questions (VizWiz) and combines it with dedicated product detection
and expiration-date reading, so answers are grounded in what's actually
detected in the image rather than guessed.

## Architecture

```
Photo + spoken question
        |
        v
  [Whisper: speech -> text]        (Phase 6)
        |
        v
  [YOLOv8: product detection] ---> [OCR: label / expiration date] (Phases 3-4)
        |                                    |
        +------------------+-----------------+
                           v
              [BLIP (LoRA fine-tuned on VizWiz): VQA]   (Phase 5)
                           |
                           v
                 [TTS: text -> speech]                  (Phase 6)
                           |
                           v
                     Spoken answer
```

## Status

Build is organized into 8 phases, tracked as we go:

1. Project setup (this step)
2. Data acquisition & inspection — VizWiz-VQA, Freiburg Groceries, ExpDate, Grozi-120
3. Product detection — YOLOv8 fine-tuned on grocery data (Colab)
4. Label/expiration-date reading — OpenCV + OCR, validated on ExpDate
5. VQA — BLIP fine-tuned on VizWiz-VQA with LoRA (Colab), tracked in W&B
6. Voice interface — Whisper (STT) + TTS
7. Pipeline integration — all components wired into one flow
8. Demo app — Streamlit/Gradio, plus this README as living documentation

## Tech stack and why

| Component | Tool | Why |
|---|---|---|
| Detection | YOLOv8 (Ultralytics) | Fast, well-documented fine-tuning API |
| OCR | OpenCV + EasyOCR/Tesseract | Standard combo; OpenCV handles preprocessing (deskew/denoise), OCR engine handles recognition |
| VQA | BLIP + LoRA (Hugging Face PEFT) | Pretrained on 129M image-text pairs already; LoRA keeps fine-tuning feasible on a free Colab GPU |
| Experiment tracking | Weights & Biases | Proof of what was tried and what worked, not just a final checkpoint |
| Voice | Whisper (STT), TTS engine | Makes this usable by the actual target user, not just a text demo |
| Demo | Streamlit/Gradio | Working UI without frontend engineering |

## Data sources

- [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/) — real photos and questions from blind users, core VQA fine-tuning set
- [Freiburg Groceries Dataset](https://github.com/PhilJd/freiburg_groceries_dataset) — product classification
- [ExpDate](https://felizang.github.io/expdate/) — expiration date recognition
- Grozi-120 / Grocery Products Dataset — product recognition

## Not in scope for v1

Real-time video processing, on-device/mobile deployment, and full
productionization (Docker, CI/CD, hosted API, monitoring) are deliberately
deferred — see project notes for the follow-on plan once v1 is proven.
