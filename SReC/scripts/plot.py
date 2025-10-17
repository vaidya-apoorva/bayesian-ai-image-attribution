import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load JSON data
with open("/mnt/ssd-data/vaidya/SReC/data_openimage.json", "r") as f:
    data = json.load(f)

# Prepare the data: Flatten all values into a single list
values = []
labels = []

for label, d_list in data.items():
    values.extend(d_list) 
    labels.extend([label] * len(d_list))  # Repeat the label for each value

# Create a DataFrame
df = pd.DataFrame({"D(l)": values, "Dataset": labels})

# Plot histogram with KDE
plt.figure(figsize=(8, 4))
sns.histplot(df, x="D(l)", hue="Dataset", bins=np.arange(-0.5, 0.5, 0.02), kde=True, edgecolor="black", palette="muted")

# Title and labels
plt.title("Histogram of D(l) Values")
plt.xlabel("D(l)")
plt.ylabel("Frequency")

plt.xticks(np.arange(-0.5, 0.5, 0.1))

# Limit the x-axis to exclude data beyond 0.2
plt.xlim(-0.5, 0.5)

# Save and show the figure
plt.savefig('/mnt/ssd-data/vaidya/SReC/openimage_histogram_d0.png', dpi=300)
plt.show()
