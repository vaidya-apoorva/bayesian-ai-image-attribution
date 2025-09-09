import os
import json
import joblib
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ----------------------------------
# Folder holding your per‐generator JSON files
json_dir = "/mnt/ssd-data/vaidya/SReC/results/json_files/coco_trained/JPEG_80"

# Where to dump your joblib'ed KDE models
kde_model_dir = "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED/coco_trained_JPEG_80/joblib"
os.makedirs(kde_model_dir, exist_ok=True)

# (Optionally) where to save a sanity‐check PNG of all KDEs
kde_plot_png = "/mnt/ssd-data/vaidya/SReC/report-results/results/KDE_plots_with_joblib/ZED/coco_trained_JPEG_80/PNG/coco_trained.png"
# ----------------------------------------------------

# STEP 1 — fit and save a KDE per JSON
for fn in os.listdir(json_dir):
    if not fn.endswith(".json"):
        continue

    name = fn.replace(".json", "")                    # e.g. "midjourneyV5_jpeg90"
    path = os.path.join(json_dir, fn)

    # load the gap list
    with open(path, "r") as f:
        data = json.load(f)
    gaps = np.array(list(data.values()), dtype=float).reshape(-1, 1)

    if gaps.shape[0] < 2:
        print(f"⚠️  skipping {name}, not enough samples")
        continue

    # fit a Gaussian KDE (you can tune bandwidth)
    kde = KernelDensity(kernel="gaussian", bandwidth=0.05)
    kde.fit(gaps)

    # save the model
    out_kde = os.path.join(kde_model_dir, f"kde_model_{name}.joblib")
    joblib.dump(kde, out_kde)
    print(f"✅ saved KDE model for {name} → {out_kde}")

# STEP 2 — (optional) plot all of them again for a visual check
plt.figure(figsize=(10,6))
x_plot = np.linspace(-0.4, 0.4, 800).reshape(-1,1)
for model_fn in sorted(os.listdir(kde_model_dir)):
    if not model_fn.endswith(".joblib"):
        continue
    name = model_fn.replace("kde_model_", "").replace(".joblib","")
    kde = joblib.load(os.path.join(kde_model_dir, model_fn))

    log_dens = kde.score_samples(x_plot)
    dens = np.exp(log_dens)
    plt.plot(x_plot[:,0], dens, label=name)

plt.xlim(-0.4,0.4)
plt.ylim(0, 45)
plt.title("OpenImages‐trained ZED Coding Cost KDEs")
plt.xlabel("Coding Cost Gap")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(kde_plot_png, dpi=300)
print(f"🖼  KDE check plot saved → {kde_plot_png}")
