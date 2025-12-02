#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kde_from_json(json_dir, output_file, file_suffix, title, xlabel):
    """Generic KDE plot for JSON value files in a directory with fixed display names."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    # Mapping from filename base to desired display label
    display_name_map = {
        "coco": "Coco",
        "firefly": "Firefly",
        "sdxl": "SD-XL",
        "dall-e3": "DALL.E3",
        "midjourneyV6": "Midj.-V6",
        "dall-e2": "DALL.E2",
        "stable_diffusion_1-5": "SD-1.5",
        "midjourneyV5": "Midj.-V5",
    }

    # Desired order of appearance
    display_order = ["Coco", "Firefly", "SD-XL", "DALL.E3", "Midj.-V6", "DALL.E2", "SD-1.5", "Midj.-V5"]

    plt.figure(figsize=(12, 6))

    # Collect valid datasets and sort according to desired order
    datasets = {}
    for filename in json_files:
        label_key = filename.replace(file_suffix, "")
        if label_key not in display_name_map:
            continue

        path = os.path.join(json_dir, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        values = list(data.values())
        if not values:
            continue

        datasets[display_name_map[label_key]] = values

    # Plot in the specified order
    for label in display_order:
        if label in datasets:
            sns.kdeplot(datasets[label], fill=True, alpha=0.5, label=label)

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Legend configuration
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.9,
        fontsize=12,
        title=None,
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    # Save both PNG and PDF
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(output_file)[0] + ".pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"KDE plot saved to: {output_file} and {pdf_path}")


if __name__ == "__main__":
    print("Creating KDE plots for SReC, RIGID, and Aeroblade...")

    base = "/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/priors"
    out_base = "/mnt/ssd-data/vaidya/gi_conference_results/bayesian_results/KDE"

    plot_kde_from_json(
        json_dir=f"{base}/SReC",
        output_file=f"{out_base}/srec.png",
        file_suffix="_d0.json",
        title="KDE of SReC D(l) Gap Values",
        xlabel="D(l) Gap Values"
    )

    plot_kde_from_json(
        json_dir=f"{base}/RIGID",
        output_file=f"{out_base}/rigid.png",
        file_suffix="_rigid_results.json",
        title="KDE of RIGID Similarity Values",
        xlabel="RIGID Similarity Values"
    )

    plot_kde_from_json(
        json_dir=f"{base}/AEROBLADE",
        output_file=f"{out_base}/aeroblade.png",
        file_suffix="_aeroblade.json",
        title="KDE of Aeroblade Distance Values",
        xlabel="Aeroblade Distance Values"
    )

    print("All KDE plots completed successfully.")
