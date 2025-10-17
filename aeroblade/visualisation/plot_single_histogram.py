import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
csv_file = "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/midjourneyv5_distances.csv"
data = pd.read_csv(csv_file)

# Group data by repo_id
grouped = data.groupby("repo_id")

# Plot the histogram
plt.figure(figsize=(12, 6))

# Define a color palette
colors = plt.cm.tab10.colors  # Use a colormap with 10 distinct colors
color_index = 0

# Plot each repo_id's distances with a different color
for repo_id, group in grouped:
    distances = group["distance"]
    plt.hist(
        distances,
        bins=30,
        alpha=0.6,
        edgecolor="black",
        label=f"{repo_id}",
        color=colors[color_index % len(colors)],  # Cycle through colors
    )
    color_index += 1

# Customize the plot
plt.title("Histogram of LPIPS Distances (Colored by Repository)", fontsize=16)
plt.xlabel("Distance", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Repositories", fontsize=10)

# Save the histogram
output_dir = "/mnt/ssd-data/vaidya/aeroblade/visualisation"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
output_file = os.path.join(output_dir, "colored_repo_distance_histogram.png")
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved histogram to {output_file}")