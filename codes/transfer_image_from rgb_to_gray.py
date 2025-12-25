import os
from PIL import Image

# ================== 可修改区域 ==================
INPUT_DIR = r"input_images"      # 输入文件夹路径
OUTPUT_DIR = r"output_images"    # 输出文件夹路径
OUTPUT_SUFFIX = "_gray"          # 输出文件名后缀
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    name, ext = os.path.splitext(fname)
    out_name = f"{name}{OUTPUT_SUFFIX}{ext}"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    img = Image.open(in_path).convert("L")  # RGB → Grayscale
    img.save(out_path)

print("Done.")
