import os
import json
import torch
import click
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src import configs, data as lc_data, network
from src.l3c import bitcoding, timer

@click.command()
@click.option("--path", type=click.Path(exists=True), help="Directory of images.")
@click.option("--file", type=click.File("r"), help="File for image names.")
@click.option("--resblocks", type=int, default=5, show_default=True, help="Number of resblocks to use.")
@click.option("--n-feats", type=int, default=64, show_default=True, help="Size of n_feats vector used.")
@click.option("--scale", type=int, default=3, show_default=True, help="Scale of downsampling")
@click.option("--load", type=click.Path(exists=True), help="Path to load model")
@click.option("--K", type=int, default=10, help="Number of clusters in logistic mixture model.")
@click.option("--crop", type=int, default=0, help="Size of image crops in training. 0 means no crop.")
@click.option("--log-likelihood", is_flag=True, default=False, help="Turn on log-likelihood calculations.")
@click.option("--save-path", type=str, help="Save directory for images.")
def main(path, file, resblocks, n_feats, scale, load, k, crop, log_likelihood, save_path):
    # Set model configurations
    configs.n_feats = n_feats
    configs.resblocks = 5
    configs.K = k
    configs.scale = scale
    configs.log_likelihood = log_likelihood
    configs.collect_probs = True

    # Load model
    checkpoint = torch.load(load)
    print(f"Loaded model from {load}.")
    print("Epoch:", checkpoint["epoch"])

    compressor = network.Compressor()
    compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()
    compressor.eval()  # Set to evaluation mode

    # Define image transformations
    transforms = []
    if crop > 0:
        transforms.insert(0, T.CenterCrop(crop))
    transform = T.Compose(transforms)

    # Load dataset
    dataset = lc_data.ImageFolder(path, [filename.strip() for filename in file], scale, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Loaded directory with {len(dataset)} images.")

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Initialize coder
    coder = bitcoding.Bitcoding(compressor)
    encoder_time_accumulator = timer.TimeAccumulator()

    # Dict to store results
    results = {}

    # Process images
    for i, (filenames, x) in enumerate(dataloader):
        assert len(filenames) == 1, filenames
        filename = filenames[0]
        file_id = filename.rsplit(".", 1)[0]  # safer split
        filepath = os.path.join(save_path, f"{file_id}.srec")

        # Delete the existing file if it exists
        if os.path.isfile(filepath):
            os.remove(filepath)

        with encoder_time_accumulator.execute():
            bits, entropy_coding_bytes, nll_list, entropy_list = coder.encode(x, filepath)

        # the last outer iteration corresponds to full resolution (level 0)
        print("Calculating D(l) for level 0...")
        print("NLL:", nll_list[-1])
        print("Entropy:", entropy_list[-1])
        D_l = nll_list[-1] - entropy_list[-1]

        full_img_path = os.path.join(path, filename)
        results[full_img_path] = D_l

    print("Encoding completed.")
    print("Results:", results)

    # Save results to JSON
    json_path = "/mnt/ssd-data/vaidya/SReC/results/json_files/coco_trained/JPEG_80"
    json_out = os.path.join(json_path, "coco_d0.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_out}")

    # Plot histogram
    if results:
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(list(results.items()), columns=["Image", "D(l)"])

        plt.figure(figsize=(8, 4))
        sns.histplot(df, x="D(l)", bins=30, kde=True, edgecolor="black")
        plt.title("Histogram of D(l) values")
        plt.xlabel("D(l)")
        plt.ylabel("Frequency")

        png_path = "/mnt/ssd-data/vaidya/SReC/results/PNG/coco_images_trained/JPEG_80"
        plot_out = os.path.join(png_path, "coco_d0.png")
        plt.savefig(plot_out, dpi=300)
        plt.show()
        print(f"Saved plot to {plot_out}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
