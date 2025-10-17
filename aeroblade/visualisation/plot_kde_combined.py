import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# List of CSV files to process
csv_files = [
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/coco_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/dalle2_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/laion_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/midjourneyv5_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/raise_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/dalle3_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/sdxl_distances.csv",
]

# Create output directory for KDE plots
output_dir = "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots/AEROBLADE/aeroblade_distance_JPEG90"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Dictionary to store distances for each repo_id across all CSVs
repo_distances = {}

# Process each CSV file
for csv_file in csv_files:
    # Load the CSV file
    print(f"Processing file: {csv_file}")
    data = pd.read_csv(csv_file)

    # Extract the base name of the CSV file (e.g., "coco_distances")
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Group data by repo_id and collect distances
    grouped = data.groupby("repo_id")
    for repo_id, group in grouped:
        if repo_id not in repo_distances:
            repo_distances[repo_id] = {}
        if csv_name not in repo_distances[repo_id]:
            repo_distances[repo_id][csv_name] = []
        repo_distances[repo_id][csv_name].extend(group["distance"].tolist())

# Generate a KDE plot for each repo_id
for repo_id, csv_data in repo_distances.items():
    plt.figure(figsize=(12, 6))

    # Plot KDE for each CSV file with a different color
    for csv_name, distances in csv_data.items():
        sns.kdeplot(
            distances,
            fill=True,  # Fill the area under the curve
            alpha=0.6,  # Transparency
            label=f"{csv_name}",  # Label for the legend
        )

    # Fixed axis ranges (adjust as needed based on your data distribution)
    x_min, x_max = -0.1, 0.1
    y_min, y_max = 0.0, 90.0

    # Customize the plot
    plt.title(f"KDE Plot of LPIPS Distances for AEROBLADE JPEG90", fontsize=16)
    plt.xlabel("Distance", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="CSV Files", fontsize=12)

    # Apply fixed axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Increase font size of axis tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save the KDE plot
    # Replace slashes in repo_id to avoid invalid file paths
    sanitized_repo_id = repo_id.replace("/", "_")
    output_file = os.path.join(output_dir, f"{sanitized_repo_id}_distance_kde.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to avoid overlapping

    print(f"Saved KDE plot for {repo_id} to {output_file}")