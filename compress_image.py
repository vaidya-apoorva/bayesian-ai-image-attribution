import os
from PIL import Image

# List all your input directories here
input_dirs = [
    "/mnt/hdd-data/vaidya/dataset/coco_resized",
    "/mnt/hdd-data/vaidya/dataset/dalle2_resized",
    "/mnt/hdd-data/vaidya/dataset/dalle3_resized",
    "/mnt/hdd-data/vaidya/dataset/laion_resized",
    "/mnt/hdd-data/vaidya/dataset/midjourneyV5_resized",
    "/mnt/hdd-data/vaidya/dataset/raise_resized",
    "/mnt/hdd-data/vaidya/dataset/sdxl_resized",
    # Add more directories if needed
]

# JPEG qualities
jpeg_qualities = [90, 80]

for input_dir in input_dirs:
    for quality in jpeg_qualities:
        output_dir = f"{input_dir}_jpeg{quality}"
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]

            try:
                with Image.open(input_path) as img:
                    rgb_img = img.convert("RGB")

                    for quality in jpeg_qualities:
                        output_dir = f"{input_dir}_jpeg{quality}"
                        output_path = os.path.join(output_dir, f"{base_name}.jpg")
                        rgb_img.save(output_path, format="JPEG", quality=quality)

            except Exception as e:
                print(f"Error converting {input_path}: {e}")

print("✅ All PNGs converted to JPEG 90 and JPEG 80 for all directories.")
