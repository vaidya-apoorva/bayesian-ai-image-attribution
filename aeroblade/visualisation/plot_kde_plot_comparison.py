import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define CSV files for each compression level
csv_files = {
    "original": [
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/original/coco_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/original/dalle2_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/original/laion_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/original/midjourneyv5_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/original/raise_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/original/dalle3_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/original/sdxl_distances.csv",
    ],
    "JPEG80": [
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG80/coco_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG80/dalle2_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG80/laion_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG80/midjourneyv5_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG80/raise_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG80/dalle3_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG80/sdxl_distances.csv",
    ],
    "JPEG90": [
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/coco_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/dalle2_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/laion_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/midjourneyv5_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/raise_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/dalle3_distances.csv",
        "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/sdxl_distances.csv",
    ],
}

# Create output directory for KDE plots
output_dir = "/mnt/ssd-data/vaidya/aeroblade/visualisation/repo_kde_plots_comparison"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store distances for each compression level
compression_distances = {"original": [], "JPEG80": [], "JPEG90": []}

# Load distances from each CSV file
for compression, files in csv_files.items():
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            if "distance" in df.columns:  # Ensure the column exists
                compression_distances[compression].extend(df["distance"].values)

# Plot KDE for each compression level
plt.figure(figsize=(10, 6))
for compression, distances in compression_distances.items():
    sns.kdeplot(distances, label=compression, fill=True, alpha=0.5)

# Add plot details
plt.title("KDE Plot Comparison: Original vs JPEG80 vs JPEG90")
plt.xlabel("Distance")
plt.ylabel("Density")
plt.legend(title="Compression Level")
plt.grid(True)

# Save the plot
output_file = os.path.join(output_dir, "kde_comparison_plot.png")
plt.savefig(output_file)
plt.show()

print(f"KDE comparison plot saved to {output_file}")