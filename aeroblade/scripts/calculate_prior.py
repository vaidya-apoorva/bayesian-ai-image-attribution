import os
import pandas as pd
import glob
import json

# --- CONFIG ---
DISTANCE_DIR = '/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/distances/'
OUTPUT_FILE = '/mnt/ssd-data/vaidya/aeroblade/aeroblade_output/distances/soft_priors.json'
EPSILON = 1e-6  # To avoid division by zero in similarity
distance_metric = 'lpips_vgg_2'

# --- STEP 1: Load and combine all CSVs ---
csv_files = glob.glob(os.path.join(DISTANCE_DIR, '*_distances.csv'))

all_distances = []

for csv in csv_files:
    df = pd.read_csv(csv)
    gen_name = os.path.basename(csv).replace('_distances.csv', '')
    df = df[df['distance_metric'] == distance_metric].copy()
    df['generator'] = gen_name
    df['filename'] = df['file'].apply(lambda x: os.path.basename(x))
    all_distances.append(df[['filename', 'generator', 'distance']])

df_all = pd.concat(all_distances)

# --- STEP 2: Convert distances to similarities ---
df_all['similarity'] = 1.0 / (df_all['distance'] + EPSILON)

# --- STEP 3: Normalize similarities per image to get soft priors ---
priors = {}

for fname, group in df_all.groupby('filename'):
    total_sim = group['similarity'].sum()
    probs = {row['generator']: row['similarity'] / total_sim for _, row in group.iterrows()}
    priors[fname] = probs

# --- STEP 4: Save priors as JSON ---
with open(OUTPUT_FILE, 'w') as f:
    json.dump(priors, f, indent=2)

print(f"✅ Soft priors saved to {OUTPUT_FILE}")
