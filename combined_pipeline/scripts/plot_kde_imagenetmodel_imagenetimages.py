import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gap_values_kde():
    """Create KDE plot for GAP_VALUES (SReC D(l) values)."""
    # Directory containing GAP_VALUES JSON files
    json_dir = "/mnt/ssd-data/vaidya/combined_pipeline/results/IMAGENET_IMAGES/SREC_IMAGENET/GAP_VALUES"
    output_dir = "/mnt/ssd-data/vaidya/combined_pipeline/results/KDE/IMAGENET_IMAGES"
    output_file = os.path.join(output_dir, "imagenetmodel_imagenetimages_srec_gap_values_kde.png")
    os.makedirs(output_dir, exist_ok=True)

    # Get all json files in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    # Generate KDE plot
    plt.figure(figsize=(12, 6))

    for filename in json_files:
        # Clean up label from filename
        if filename.startswith("Imagenet256-"):
            label = filename.replace("Imagenet256-", "imagenet256-").replace("_d0.json", "")
        else:
            label = filename.replace("_d0.json", "")

        json_path = os.path.join(json_dir, filename)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract only the gap values (ignore the image paths)
        gaps = list(data.values())

        if len(gaps) == 0:
            continue  # Skip if no data

        sns.kdeplot(gaps, fill=True, alpha=0.5, label=label)

    plt.title("KDE of SReC D(l) Gap Values (ImageNet)", fontsize=16)
    plt.xlabel("D(l) Gap Values", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Dataset", fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

    # Increase font size of axis tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("GAP VALUES KDE plot saved to: {}".format(output_file))


def plot_similarity_values_kde():
    """Create KDE plot for SIMILARITY_VALUES (RIGID similarity scores)."""
    # Directory containing SIMILARITY_VALUES JSON files
    json_dir = "/mnt/ssd-data/vaidya/combined_pipeline/results/IMAGENET_IMAGES/RIGID_IMAGENET/SIMILARITY_VALUES"
    output_dir = "/mnt/ssd-data/vaidya/combined_pipeline/results/KDE/IMAGENET_IMAGES"
    output_file = os.path.join(output_dir, "imagenetmodel_imagenetimages_rigid_similarity_values_kde.png")
    os.makedirs(output_dir, exist_ok=True)

    # Get all json files in the directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    # Generate KDE plot
    plt.figure(figsize=(12, 6))

    for filename in json_files:
        # Clean up label from filename
        label = filename.replace("_rigid_results.json", "")
        json_path = os.path.join(json_dir, filename)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract only the similarity values (ignore the image paths)
        similarities = list(data.values())

        if len(similarities) == 0:
            continue  # Skip if no data

        sns.kdeplot(similarities, fill=True, alpha=0.5, label=label)

    plt.title("KDE of RIGID Similarity Values (ImageNet)", fontsize=16)
    plt.xlabel("RIGID Similarity Values", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Dataset", fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

    # Increase font size of axis tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("SIMILARITY VALUES KDE plot saved to: {}".format(output_file))


# Create both plots
if __name__ == "__main__":
    print("Creating KDE plots for GAP_VALUES and SIMILARITY_VALUES...")
    plot_gap_values_kde()
    plot_similarity_values_kde()
    print("Both KDE plots completed!")
