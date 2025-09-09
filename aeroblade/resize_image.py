import os
from PIL import Image

# Your actual image directory
input_dir = "/mnt/hdd-data/vaidya/dataset/sdxl"
output_dir = input_dir + "_resized"
resize_to = (512, 512)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Track sizes
size_counts = {}
image_paths = []

# Collect image info
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        full_path = os.path.join(input_dir, filename)
        try:
            with Image.open(full_path) as img:
                size = img.size
                size_counts[size] = size_counts.get(size, 0) + 1
                image_paths.append((filename, size))
        except Exception as e:
            print(f"Failed to open {filename}: {e}")

# Show image size stats
print("\nImage size distribution:")
for size, count in size_counts.items():
    print(f"  {size}: {count} image(s)")

# Resize if needed
if len(size_counts) > 1 or list(size_counts.keys())[0] != resize_to:
    print(f"\nResizing images to {resize_to} into: {output_dir}")
    for filename, _ in image_paths:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        try:
            with Image.open(input_path) as img:
                resized = img.resize(resize_to, Image.LANCZOS)
                resized.save(output_path)
        except Exception as e:
            print(f"Error resizing {filename}: {e}")
    print("✅ All images resized.")
else:
    print("\n✅ All images already have correct dimensions. No resizing done.")

