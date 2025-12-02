
---

```markdown
#  Baseline Classifier — Real vs AI (Diffusion) & Model ID

This project implements a **two-level image forensics baseline** for detecting AI-generated (diffusion-based) images and identifying their source models.

---

##  Overview

| Level | Description | Classes | Example Accuracy |
|:------|:-------------|:---------|:-----------------|
| **Level-1** | Real vs AI | 2 (Real, AI) | ~0.92–0.95 |
| **Level-2** | Diffusion Model Identification | 7 (DALLE2, DALLE3, Firefly, MidjourneyV5, MidjourneyV6, SDXL, Stable Diffusion 1.5) | ~0.74–0.82 |

This setup is **intentionally simple** — based on ResNet-34, moderate augmentations, and conservative hyperparameters — designed to serve as a *baseline*, not an optimized system.

---

##  Directory Structure

### Project
```

/mnt/ssd-data/vaidya/baseline_classifier
├── two_level_diffusion_forensics.py   # Training + inference script
├── README.md                          # This file
└── runs/                              # Model checkpoints & logs

```

### Dataset
```

/mnt/hdd-data/vaidya/dataset
├── coco/                    # Real
├── imagenet/                # Real
├── dalle2/                  # AI
├── dalle3/                  # AI
├── firefly/                 # AI
├── midjourneyV5/            # AI
├── midjourneyV6/            # AI
├── sdxl/                    # AI
├── stable_diffusion_1_5/    # AI
└── level{1,2}_{train,val,test}.csv

````

---

##  Environment Setup

**CUDA Driver:** 470.xx  
**CUDA Version:** 11.4  
**Python:** 3.10  
**PyTorch:** 1.12.1 (cu113)

### Create the environment
```bash
conda create -n baseline python=3.10 -y
conda activate baseline
````

### Install PyTorch and dependencies

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --index-url https://download.pytorch.org/whl/cu113

pip install numpy==1.24.4 pillow==9.5.0 scikit-learn==1.2.2 tqdm pandas matplotlib
```

### Verify installation

```bash
python -c "import torch;print(torch.__version__);print('CUDA:',torch.cuda.is_available());print('GPUs:',torch.cuda.device_count())"
```

---

## 🧩 Dataset CSV Creation (80/10/10 split)

From:

```
/mnt/hdd-data/vaidya/dataset
```

Run:

```bash
python - <<'PY'
import csv, random
from pathlib import Path

root = Path("/mnt/hdd-data/vaidya/dataset")
reals = ["coco", "imagenet"]
ais = ["dalle2","dalle3","firefly","midjourneyV5","midjourneyV6","sdxl","stable_diffusion_1_5"]
exts = {".jpg",".jpeg",".png",".webp",".bmp",".JPG",".JPEG",".PNG"}

def ls(d): return [str(p) for p in (root/d).rglob("*") if p.suffix in exts]

# Build level1.csv (0 = real, 1 = AI)
with open("level1.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["path","label"])
    [w.writerow([p,0]) for d in reals for p in ls(d)]
    [w.writerow([p,1]) for d in ais for p in ls(d)]

# Build level2.csv (7-way diffusion model)
with open("level2.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["path","label"])
    for i,d in enumerate(ais): [w.writerow([p,i]) for p in ls(d)]

# Split 80/10/10
def split(inf):
    rows=list(csv.DictReader(open(inf)))
    random.Random(42).shuffle(rows)
    n=len(rows); tr=n*80//100; va=n*10//100; te=n-tr-va
    base=Path(inf).stem
    for name, subset in [("train",rows[:tr]),("val",rows[tr:tr+va]),("test",rows[tr+va:])]:
        out=f"{base}_{name}.csv"
        with open(out,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["path","label"])
            w.writeheader(); w.writerows(subset)
        print(out,len(subset))
split("level1.csv"); split("level2.csv")
PY
```

You should see:

```
level1_train.csv, level1_val.csv, level1_test.csv
level2_train.csv, level2_val.csv, level2_test.csv
```

---

## 🚀 Training

### Level-1 (Real vs AI)

```bash
python two_level_diffusion_forensics.py --mode train_level1 \
  --csv /mnt/hdd-data/vaidya/dataset/level1_train.csv \
  --out runs/level1 --epochs 20 --batch-size 64 \
  --val-split 0.1 --robust-augs --balanced-sampler
```

### Level-2 (Diffusion Model ID)

```bash
python two_level_diffusion_forensics.py --mode train_level2 \
  --csv /mnt/hdd-data/vaidya/dataset/level2_train.csv \
  --out runs/level2 --epochs 30 --batch-size 64 --val-split 0.1 \
  --robust-augs \
  --class-names DALLE2 DALLE3 FIREFLY MIDJ_V5 MIDJ_V6 SDXL SD15
```

Checkpoints will be saved under:

```
runs/<level>/best.pt
runs/<level>/last.pt
```

---

## 🧪 Evaluation

### Level-1 Evaluation

```bash
python - <<'PY'
import torch
from two_level_diffusion_forensics import SimpleImageCSVDataset, ResNetClassifier, evaluate
from torch.utils.data import DataLoader
csv="/mnt/hdd-data/vaidya/dataset/level1_test.csv"
ckpt="runs/level1/best.pt"
ds=SimpleImageCSVDataset(csv,is_train=False,img_size=224)
ld=DataLoader(ds,64,shuffle=False,num_workers=4,pin_memory=True)
m=ResNetClassifier(2,pretrained=False).cuda()
m.load_state_dict(torch.load(ckpt,map_location="cuda")["model"])
crit=torch.nn.CrossEntropyLoss()
loss,acc,auc,cm=evaluate(m,ld,torch.device("cuda"),crit,2,is_binary=True)
print("Level-1 TEST | acc:",round(acc,4),"AUC:",round(auc,4))
print("Confusion Matrix:\n",cm)
PY
```

### Level-2 Evaluation + Confusion Matrix Plot

```bash
python - <<'PY'
import torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from two_level_diffusion_forensics import SimpleImageCSVDataset, ResNetClassifier, evaluate
from torch.utils.data import DataLoader

names=["DALLE2","DALLE3","FIREFLY","MIDJ_V5","MIDJ_V6","SDXL","SD15"]
csv="/mnt/hdd-data/vaidya/dataset/level2_test.csv"
ckpt="runs/level2/best.pt"

ds=SimpleImageCSVDataset(csv,is_train=False,img_size=224)
ld=DataLoader(ds,64,shuffle=False,num_workers=4,pin_memory=True)
m=ResNetClassifier(len(names),pretrained=False).cuda()
m.load_state_dict(torch.load(ckpt,map_location="cuda")["model"])
crit=torch.nn.CrossEntropyLoss()
loss,acc,_,cm=evaluate(m,ld,torch.device("cuda"),crit,len(names),is_binary=False)

print("Level-2 TEST | acc:",round(acc,4))
print("Confusion Matrix:\n",cm)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=names)
fig,ax=plt.subplots(figsize=(8,8))
disp.plot(ax=ax,cmap="Blues",colorbar=True,xticks_rotation=45)
plt.title(f"Level-2 Confusion Matrix (acc={acc:.3f})")
plt.tight_layout()
plt.savefig("level2_confusion_matrix.png",dpi=300)
print("Saved → level2_confusion_matrix.png")
PY
```

---

## 🔍 Inference

Run a single image through the two-level pipeline:

```bash
python two_level_diffusion_forensics.py --mode infer \
  --img /path/to/image.jpg \
  --ckpt-level1 runs/level1/best.pt \
  --ckpt-level2 runs/level2/best.pt \
  --class-names DALLE2 DALLE3 FIREFLY MIDJ_V5 MIDJ_V6 SDXL SD15
```

Outputs:

```
p_real, p_ai, predicted_class (if AI)
```

---
```
Would you like me to generate a small `build_csvs.py` utility file (to accompany this README) so you can rebuild splits with one command instead of the long inline script?
```
