import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import label2rgb
import argparse

# --- Parameters ---
parser = argparse.ArgumentParser()
parser.add_argument('folder', help = "Type one folder between 'test', 'valid' or 'train'")
args = parser.parse_args()

directory = "tusimple_processed_split" ## Modify here with the original directory where you saved TuSimple data!!!
masks_folder = os.path.join(directory, args.folder, "masks_multiclass")
img_folder = os.path.join(directory, args.folder, "images")
save_folder = os.path.join(directory, args.folder, "masks_assigned")

os.makedirs(save_folder, exist_ok=True)

class_options = ['continuous', 'dashed', 'unmarked']
class_map = {name: i+1 for i, name in enumerate(class_options)}  # 1,2,3
processed_files = os.listdir(save_folder)

for fname in sorted(os.listdir(masks_folder)):
    if not fname.endswith((".png", ".jpg", ".tif")):
        continue
    
    if fname in processed_files:
        continue

    mask_path = os.path.join(masks_folder, fname)
    save_path = os.path.join(save_folder, fname)
    img_path = os.path.join(img_folder, fname)  # assumes same name
    if not os.path.exists(img_path):
        print(f"No matching image for {fname}, skipping...")
        continue

    # Load mask and original image
    instance_mask_full = imread(mask_path)
    original_img = imread(img_path)
    instance_mask_full = instance_mask_full.astype(int)

    # Unique labels
    labels = [l for l in np.unique(instance_mask_full) if l != 0]

    # Create an empty relabel mask
    relabeled = np.zeros_like(instance_mask_full)

    print(f"\nProcessing {fname} - found {len(labels)} lane instances")

    # Interactive labeling
    for l in labels:
        mask = (instance_mask_full == l)
        overlay = label2rgb(mask, image=original_img, bg_label=0, alpha=0.5)

        plt.imshow(overlay)
        plt.title(f"{fname} - Lane {l}")
        plt.axis('off')
        plt.show(block=False)

        key = input("Assign class (1=continuous, 2=dashed, 3=unmarked, 0=skip): ")
        plt.close()

        if key in ["1", "2", "3"]:
            relabeled[mask] = int(key)

    # Save relabeled mask with same name
    imsave(save_path, relabeled.astype(np.uint8))
    print(f"Saved relabeled mask: {save_path}")
