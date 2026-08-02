import os
import subprocess
import sys
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

TESSDATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tessdata")

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <image_path> [prompt]")
    sys.exit(1)

image = sys.argv[1]
base = os.path.splitext(os.path.basename(image))[0]
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

prompt = sys.argv[2] if len(sys.argv) > 2 else (
    "Extract all text found on the image, including handwritten signatures."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

torch.cuda.empty_cache()
generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
qwen_output = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

qwen_path = os.path.join(out_dir, f"{base}.txt")
with open(qwen_path, "w", encoding="utf-8") as f:
    f.write(qwen_output)
print(f"Qwen  -> {qwen_path}")

tess_path = os.path.join(out_dir, f"{base}_tess.txt")
env = os.environ.copy()
env["TESSDATA_PREFIX"] = os.path.abspath(TESSDATA_DIR)
try:
    result = subprocess.run(
        ["tesseract", image, "stdout", "-l", "grc_hist+ell", "--psm", "6"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    tess_text = result.stdout.strip()
    if tess_text:
        with open(tess_path, "w", encoding="utf-8") as f:
            f.write(tess_text)
        print(f"Tess  -> {tess_path}")
    else:
        print(f"Tesseract produced no output. stderr: {result.stderr.strip()}")
except Exception as e:
    print(f"Tesseract failed: {e}")
