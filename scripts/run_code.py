from data import load_multiclassdata
import pandas as pd
import torch
import multi_classes
from torch import nn
import os
from tqdm import tqdm
import numpy as np

# 1. Reading data and checking balance

data_metrics = []
train_df, valid_df, test_df = load_multiclassdata()

red_train_df = train_df[:len(train_df) // 2].copy()
data_metrics.append({
    'dataset': "red_train_df",
    'len': len(red_train_df),
    'background': red_train_df['background'].mean(),
    'continuous': red_train_df['1_continuous'].mean(),
    'dashed': red_train_df['2_dashed'].mean(),
    'unmarked': red_train_df['3_undefined'].mean()
})

red_valid_df = valid_df[:len(valid_df) // 2].copy()
data_metrics.append({
    'dataset': "red_valid_df",
    'len': len(red_valid_df),
    'background': red_valid_df['background'].mean(),
    'continuous': red_valid_df['1_continuous'].mean(),
    'dashed': red_valid_df['2_dashed'].mean(),
    'unmarked': red_valid_df['3_undefined'].mean()
})

red_test_df = test_df[:len(test_df) // 2].copy()
data_metrics.append({
    'dataset': "red_test_df",
    'len': len(red_test_df),
    'background': red_test_df['background'].mean(),
    'continuous': red_test_df['1_continuous'].mean(),
    'dashed': red_test_df['2_dashed'].mean(),
    'unmarked': red_test_df['3_undefined'].mean()
})
print(pd.DataFrame(data_metrics))

# 2. Build datasets and loaders
train_dataset = multi_classes.NoisyLanesJointDataset(train_df, byol_aug=multi_classes.get_augmentation(), seg_aug=multi_classes.get_augmentation(), to_tensor=True, resize = 0.4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, generator = torch.Generator().manual_seed(2))

valid_dataset = multi_classes.NoisyLanesJointDataset(valid_df, byol_aug=multi_classes.get_augmentation(), to_tensor=True, resize = 0.4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=12, shuffle=False)

test_dataset = multi_classes.NoisyLanesJointDataset(test_df, byol_aug=multi_classes.get_augmentation(), to_tensor=True, resize = 0.4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle = False)

# 3. Define model and hyperparameters
model = multi_classes.BYOL_LinkNet(n_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
seg_loss = nn.CrossEntropyLoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "E:\\PhD_NTUA\\5_Projects\\2_AttributesExtraction\\Intersection\\TuSimple_MC_v2"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "lambda2.pth")

# 4. Train model
train_logs, valid_logs = multi_classes.train_model(model=model,
            train_loader=train_loader,
            val_loader = valid_loader,
            optimizer = optimizer,
            byol_loss=multi_classes.byol_loss,
            segmentation_loss=seg_loss,
            lambda_seg = 0.25,
            lambda_ssl = 0.75,
            device = DEVICE,
            epochs = 100,
            early_stopping = 20,
            save_path = os.path.join(save_dir, "lambda2.pth"),
            visualize = True)

pd.DataFrame(train_logs).to_csv(save_path.replace(".pth", "_trainLogs.csv"))
pd.DataFrame(valid_logs).to_csv(save_path.replace(".pth", "_validLogs.csv"))

# 5. Test model
# 5.1 Choose test dataset and upload best model
# test_dataset = multi_classes.NoisyLanesJointDataset(test_df, to_tensor=True, byol_aug=multi_classes.get_augmentation())
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=True)

model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

class_names = ['background', 'continuous', 'dashed', 'unmarked']

# 5.2 Average metrics
all_metrics = []
    
with torch.no_grad():
        multi_classes.visualize_pca_and_segmentation(model, test_loader, DEVICE, tag="Test", max_batches=3)
        
        progress = tqdm(test_loader, desc="Calculating Metrics", leave=True)
        for _, _, img, msk in progress:
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            preds = model.forward_segmentation(img)

            batch_metrics = multi_classes.calculate_metrics_multiclass(preds, msk, num_classes=4, reduction='macro')
            all_metrics.append(batch_metrics)

            # Show live metric on progress bar
            progress.set_postfix({
                "F1": f"{batch_metrics['f1score']:.3f}",
                "IoU": f"{batch_metrics['iou']:.3f}",
            })

# Convert to DataFrame
df_metrics = pd.DataFrame(all_metrics)

# Show average results
print("\n=== Results ===")
print("Average Metrics (macro-reduced):")
print(df_metrics.mean().to_frame("Value"))

# 5.3 Calculate average per class for each metric
all_metrics = []
    
with torch.no_grad():
        multi_classes.visualize_pca_and_segmentation(model, test_loader, DEVICE, tag="Test", max_batches=3)
        
        progress = tqdm(test_loader, desc="Calculating Metrics", leave=True)
        for _, _, img, msk in progress:
            img, msk = img.to(DEVICE), msk.to(DEVICE)
            preds = model.forward_segmentation(img)

            batch_metrics = multi_classes.calculate_metrics_multiclass(preds, msk, num_classes=4, reduction='none')
            all_metrics.append(batch_metrics)

            # Show live metric on progress bar
            progress.set_postfix({
                "F1": f"{batch_metrics['f1score'].mean():.3f}",
                "IoU": f"{batch_metrics['iou'].mean():.3f}",
            })

# Convert to DataFrame
df_metrics = pd.DataFrame(all_metrics)

average_metrics = {}

for metric in df_metrics.columns:
    # First, flatten all batches across all rows
    all_batches = []
    for row_batches in df_metrics[metric]:
        all_batches.extend(row_batches)  # Add all 4 batches from this row
    
    # Now all_batches has 4484 batches, each with 4 class values
    # Transpose to get classes as rows
    transposed = list(zip(*all_batches))
    
    # Calculate mean for each class across all 4484 batches
    average_metrics[metric] = [np.mean(class_values) for class_values in transposed]

# Convert to DataFrame
average_df = pd.DataFrame(average_metrics, index=[class_names[i] for i in range(4)])

print("Per-class Metrics:")
print(average_df)