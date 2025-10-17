import torch
from aeroblade.distances import distance_from_config

# Create dummy datasets as tensors or a minimal dataset class
dummy_img1 = torch.rand(1, 3, 224, 224)
dummy_img2 = dummy_img1 + torch.randn_like(dummy_img1)*0.1

# Wrap into dataset-like structure if required or pass tensors if supported

dist_metric = distance_from_config("lpips_vgg_2").compute

# Check if distance is non-zero
print(dist_metric(ds_a=[(dummy_img1, None)], ds_b=[(dummy_img2, None)]))

