import pandas as pd
import matplotlib.pyplot as plt
import os

# List of CSV files to process
csv_files = [
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/coco_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/dalle2_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/laion_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/midjourneyv5_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/raise_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/dalle3_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/sdxl_distances.csv",
]

# Create output directory for histograms
output_dir = "/mnt/ssd-data/vaidya/aeroblade/visualisation/repo_histograms"
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

# Generate a single histogram for each repo_id
for repo_id, csv_data in repo_distances.items():
    plt.figure(figsize=(12, 6))

    # Define a color palette
    colors = plt.cm.tab10.colors  # Use a colormap with 10 distinct colors
    color_index = 0

    # Plot distances for each CSV file with a different color
    for csv_name, distances in csv_data.items():
        plt.hist(
            distances,
            bins=30,
            alpha=0.6,
            edgecolor="black",
            label=f"{csv_name}",
            color=colors[color_index % len(colors)],  # Cycle through colors
        )
        color_index += 1

    # Customize the plot
    plt.title(f"Histogram of LPIPS Distances for {repo_id}", fontsize=16)
    plt.xlabel("Distance", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="CSV Files", fontsize=10)

    # Save the histogram
    # Replace slashes in repo_id to avoid invalid file paths
    sanitized_repo_id = repo_id.replace("/", "_")
    output_file = os.path.join(output_dir, f"{sanitized_repo_id}_distance_histogram.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to avoid overlapping

    print(f"Saved combined histogram for {repo_id} to {output_file}")