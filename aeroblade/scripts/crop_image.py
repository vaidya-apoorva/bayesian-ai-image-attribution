import os
from PIL import Image

# Input/output directories
input_dir = "/mnt/hdd-data/vaidya/dataset/real"
cropped_dir = "/mnt/hdd-data/vaidya/dataset/laion/cropped_images"
output_dir = cropped_dir + "_cropped"
crop_to = (384, 384)

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Crop images
print(f"Cropping images to {crop_to} and saving to: {output_dir}\n")
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                width, height = img.size
                if width < crop_to[0] or height < crop_to[1]:
                    print(f"Skipping {filename}: too small to crop to {crop_to}")
                    continue

                # Center-crop
                left = (width - crop_to[0]) // 2
                top = (height - crop_to[1]) // 2
                right = left + crop_to[0]
                bottom = top + crop_to[1]

                cropped = img.crop((left, top, right, bottom))
                cropped.save(output_path)
        except Exception as e:
            print(f"Error cropping {filename}: {e}")

print("\n Cropping completed.")