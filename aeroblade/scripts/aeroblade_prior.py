import os
import pandas as pd
import glob
import json

# --- CONFIG ---
DISTANCE_DIR = '/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/distances/'
OUTPUT_FILE = os.path.join(DISTANCE_DIR, 'soft_priors.json')
EPSILON = 1e-6  # To avoid division by zero in similarity
DISTANCE_METRIC = 'lpips_vgg_2'

# --- STEP 1: Load and combine all CSVs ---
csv_files = glob.glob(os.path.join(DISTANCE_DIR, '*_distances.csv'))

all_distances = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    generator_name = os.path.basename(csv_file).replace('_distances.csv', '')
    # Filter for selected distance metric only
    df = df[df['distance_metric'] == DISTANCE_METRIC].copy()
    if df.empty:
        continue  # skip files with no matching rows
    # Extract filename only (basename)
    df['filename'] = df['file'].apply(os.path.basename)
    df['generator'] = generator_name
    all_distances.append(df[['filename', 'generator', 'distance']])

if not all_distances:
    raise ValueError("No distance data found for metric '{}'".format(DISTANCE_METRIC))

df_all = pd.concat(all_distances, ignore_index=True)

# --- STEP 2: Convert distances to similarities ---
df_all['similarity'] = 1.0 / (df_all['distance'] + EPSILON)

# --- STEP 3: Normalize similarities per image to get soft priors ---
priors = {}

for fname, group in df_all.groupby('filename'):
    total_similarity = group['similarity'].sum()
    # Normalize similarity to get probability distribution
    probs = {row['generator']: row['similarity'] / total_similarity for _, row in group.iterrows()}
    priors[fname] = probs

# --- STEP 4: Save priors as JSON ---
with open(OUTPUT_FILE, 'w') as f:
    json.dump(priors, f, indent=2)

print(f"SUCCESS: Soft priors saved to {OUTPUT_FILE}")
