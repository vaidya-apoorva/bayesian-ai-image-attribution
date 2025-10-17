import os
import csv
import json

csv_dir = "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/distances"

image_distances = {}

for filename in os.listdir(csv_dir):
    if filename.endswith("_distances.csv"):
        csv_path = os.path.join(csv_dir, filename)
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("repo_id") == "max":
                    file_key = row.get("file")
                    distance_value = float(row.get("distance"))
                    image_distances[file_key] = {
                        "distance": distance_value
                    }

output_path = "image_distances.json"
with open(output_path, "w") as jf:
    json.dump(image_distances, jf, indent=2)

print(f"Saved {len(image_distances)} entries to {output_path}")
