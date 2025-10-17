import os
import json

# Directory containing *_d0.json files
json_dir = "/mnt/ssd-data/vaidya/SReC/results/json_files/openimages_trained/JPEG_80"
output_path = "/mnt/ssd-data/vaidya/SReC/results/image_coding_cost_open_images_jpeg80.json"

image_gaps = {}

# Process each *_d0.json file
for filename in os.listdir(json_dir):
    if filename.endswith("_d0.json"):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, "r") as f:
            data = json.load(f)
            for full_path, gap in data.items():
                file_name = os.path.basename(full_path)
                image_gaps[file_name] = {
                    "gap": gap
                }

# Save to output JSON
with open(output_path, "w") as out_file:
    json.dump(image_gaps, out_file, indent=2)

print(f"Saved {len(image_gaps)} entries to {output_path}")
