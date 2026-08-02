# QwenOCR

Dual OCR pipeline: Qwen2.5-VL for general text extraction + Tesseract fallback for polytonic Greek.

## Requirements

- Python 3.12
- NVIDIA GPU with 8+ GB VRAM
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
uv sync
```

If Tesseract Greek language data is not installed system-wide, download it:

```bash
mkdir -p tessdata
curl -sL "https://github.com/tesseract-ocr/tessdata/raw/main/grc_hist.traineddata" -o tessdata/grc_hist.traineddata
curl -sL "https://github.com/tesseract-ocr/tessdata/raw/main/grc.traineddata" -o tessdata/grc.traineddata
cp /usr/share/tesseract-ocr/5/tessdata/ell.traineddata tessdata/
cp /usr/share/tesseract-ocr/5/tessdata/eng.traineddata tessdata/
cp /usr/share/tesseract-ocr/5/tessdata/osd.traineddata tessdata/
```

## Usage

```bash
.venv/bin/python scripts/run_ocr.py <image_path> [prompt]
```

Outputs:

- `output/<image>.txt` — Qwen extraction (layout, handwriting, mixed content)
- `output/<image>_tess.txt` — Tesseract extraction (polytonic Greek reference)

## Example

```bash
.venv/bin/python scripts/run_ocr.py ./images/manuscript.jpg
.venv/bin/python scripts/run_ocr.py ./images/manuscript.jpg "Extract only the main text body"
```
