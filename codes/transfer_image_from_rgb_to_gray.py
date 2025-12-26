##
# This script converts all RGB images in the input directory to grayscale
# and saves them to the output directory with a specified suffix.
##



import os
from PIL import Image, ImageOps


# ================== settings ==================
INPUT_DIR = r"E:\京大\visual_interface\tower_experiment\picture"    # input folder path
OUTPUT_DIR = r"E:\京大\visual_interface\tower_experiment\picture"    # output folder path
OUTPUT_SUFFIX = "_gray"          # output file name suffix
# ===============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    name, ext = os.path.splitext(fname)
    out_name = f"{name}{OUTPUT_SUFFIX}{ext}"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    img = Image.open(in_path)
    img = ImageOps.exif_transpose(img)  # rotate according to EXIF
    img = img.convert("L")              # convert to grayscale

    img.save(out_path)

print("Done.")