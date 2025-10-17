import os
import shutil
import random

# Base paths
src_base = "/mnt/hdd-data/apoorva/dataset"
dst_base = "/mnt/hdd-data/vaidya/dataset/test_set/"
selected_folders = ["dalle3/png/"]  # <-- update this list as needed
num_images = 200
allowed_exts = (".jpg", ".jpeg", ".png")

os.makedirs(dst_base, exist_ok=True)

for folder in selected_folders:
    src_folder = os.path.join(src_base, folder)
    dst_folder = os.path.join(dst_base, folder)

    if not os.path.isdir(src_folder):
        print(f"[SKIP] Not a directory: {src_folder}")
        continue

    # List image files
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(allowed_exts)]
    if len(image_files) == 0:
        print(f"[SKIP] No image files in: {src_folder}")
        continue

    # Select up to 10 randomly
    selected = random.sample(image_files, min(num_images, len(image_files)))
    os.makedirs(dst_folder, exist_ok=True)

    for filename in selected:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.copy2(src_path, dst_path)

    print(f"[INFO] Copied {len(selected)} images from '{folder}'")

print("[DONE] Subset created.")
