import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KernelDensity
import joblib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Mapping CSV names to generator groups
csv_mapping = {
    "coco_distances": "Real",
    "raise_distances": "Real",
    "dalle2_distances": "DALL-E",
    "dalle3_distances": "DALL-E",
    "midjourneyv5_distances": "MidJourney",
    "sdxl_distances": "StableDiffusion",
    "laion_distances": "StableDiffusion",
}

# Input CSV files
csv_files = [
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/coco_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/dalle2_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/laion_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/midjourneyv5_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/raise_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/dalle3_distances.csv",
    "/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/JPEG90/sdxl_distances.csv",
]

# Output directories
kde_model_dir = "/mnt/ssd-data/vaidya/aeroblade/models/generator_kdes"
vis_dir = "/mnt/ssd-data/vaidya/aeroblade/visualisation"
os.makedirs(kde_model_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Collect distances per generator
generator_distances = {}

for csv_path in csv_files:
    name = os.path.splitext(os.path.basename(csv_path))[0]
    generator = csv_mapping.get(name)
    if generator is None:
        continue

    df = pd.read_csv(csv_path)
    distances = df["distance"].tolist()

    if generator not in generator_distances:
        generator_distances[generator] = []

    generator_distances[generator].extend(distances)

# Train KDEs and save visualizations
bandwidth = 0.02
kde_models = {}

for gen, dists in generator_distances.items():
    X = np.array(dists).reshape(-1, 1)

    # Train KDE
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(X)
    kde_models[gen] = kde

    # Save model
    model_path = os.path.join(kde_model_dir, f"kde_model_{gen}.joblib")
    joblib.dump(kde, model_path)

    # Generate density plot using scipy KDE for smoother curve
    plt.figure(figsize=(8, 4))
    xs = np.linspace(min(dists), max(dists), 1000)
    kde_smooth = gaussian_kde(dists, bw_method=bandwidth / np.std(dists))
    plt.plot(xs, kde_smooth(xs), label=gen)
    plt.fill_between(xs, kde_smooth(xs), alpha=0.3)
    plt.title(f"KDE for {gen}")
    plt.xlabel("LPIPS Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # Save PNG
    png_path = os.path.join(vis_dir, f"kde_{gen}.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Saved KDE model and plot for {gen}")
