import cv2
import torch
from skimage import io
import albumentations as album
from torch.nn import functional as F
import segmentation_models_pytorch as smp
from skimage.transform import resize
from torch import nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import random
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
from tqdm import tqdm
import os

def get_augmentation():
    transform = album.Compose([
        # album.Resize(720, 1280, interpolation=cv2.INTER_LINEAR),
        album.HorizontalFlip(p=0.5),
        album.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        album.ToGray(p=0.2),
        album.GaussianBlur(blur_limit=(3, 7), p=0.1),
        album.Solarize(p=0.05),
        album.CoarseDropout(p=1, num_holes_range=(4,4), hole_height_range=[16, 16], hole_width_range=[16, 16])
    ])
    return transform
    
class NoisyLanesJointDataset(torch.utils.data.Dataset):
    def __init__(self, df, byol_aug, seg_aug=None, to_tensor=True, resize = None):
        self.df = df.reset_index(drop=True)
        self.byol_aug = byol_aug
        self.seg_aug = seg_aug
        self.to_tensor = to_tensor
        self.resize = resize

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(row['directory'], 'images', row['name'])
        mask_path = os.path.join(row['directory'], 'masks', row['name'])

        img = io.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Problem reading image {img_path}")

        msk = io.imread(mask_path)
        if msk is None:
            raise FileNotFoundError(f"Problem reading mask {mask_path}")
        
        if self.resize:
            height, width = img.shape[:2]
            new_height = int(height * self.resize)
            new_width = int(width * self.resize)

            img = resize(
                img, (new_height, new_width),
                order=1,  # bilinear interpolation
                preserve_range=True,
                anti_aliasing=True
            ).astype(img.dtype)

            msk = resize(
                msk, (new_height, new_width),
                order=0,  # nearest
                preserve_range=True,
                anti_aliasing=False
            ).astype(msk.dtype)

        # BYOL augmentations (two views)
        view1 = self.byol_aug(image=img)['image']
        view2 = self.byol_aug(image=img)['image']

        # Segmentation augmentation (optional, normally milder)
        if self.seg_aug:
            seg_sample = self.seg_aug(image=img, mask=msk)
            img_seg, msk = seg_sample['image'], seg_sample['mask']
        else:
            img_seg = img

        if self.to_tensor:
            # Normalize BYOL views to tensors
            view1 = torch.tensor(view1).permute(2, 0, 1).float() / 255.0
            view2 = torch.tensor(view2).permute(2, 0, 1).float() / 255.0

            # Segmentation tensors
            img_seg = torch.tensor(img_seg).permute(2, 0, 1).float() / 255.0
            msk = torch.tensor(msk).long() # This had to change for multiclass segmentation: (H, W), integer labels

        return view1, view2, img_seg, msk

    def __len__(self):
        return len(self.df)

def byol_loss(z1, p2, z2, p1):
    # Normalize the representations
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)

    # Symmetric loss
    loss = 2 - 2 * (F.cosine_similarity(p1, z2.detach(), dim=-1).mean() + 
                    F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2
    return loss

class BYOL_LinkNet(nn.Module):
    def __init__(self, projection_dim=256, hidden_dim=4096, encoder_name = "resnet34", n_classes = 1):
        super().__init__()
        self.n_classes = n_classes
        act_function = 'sigmoid' if n_classes == 1 else None

        # Full LinkNet backbone from SMP
        self.linknet = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=self.n_classes,
            encoder_depth=4,
            activation = act_function
        )

        # Extract encoder and decoder
        self.encoder = self.linknet.encoder
        self.decoder = self.linknet.decoder
        self.segmentation_head = self.linknet.segmentation_head

        # Get encoder output dim
        self.encoder_output_dim = self.encoder.out_channels[-1]

        # BYOL projector head
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # BYOL predictor head
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward_encoder(self, x):
        feats = self.encoder(x)[-1]  # Get last encoder feature map
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        feats = feats.view(feats.size(0), -1)
        return feats

    def forward_byol(self, x):
        feats = self.forward_encoder(x)
        z = self.projector(feats)
        p = self.predictor(z)
        return z, p

    def forward_segmentation(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        mask = self.segmentation_head(decoder_output) # logits (B, n_classes, H, W)
        return mask

def visualize_pca_and_segmentation(model, loader, device, tag="Train", max_batches=None):
    model.eval()
    all_z, all_imgs, all_preds, all_masks = [], [], [], []

    # Define fixed colormap (0=black, 1=red, 2=green, 3=blue as example)
    cmap = ListedColormap([
        (0.0, 0.0, 0.0),   # 0 background = black
        (0.0, 0.0, 1.0),   # 1 continuous = blue
        (1.0, 0.0, 0.0),   # 2 dashed = red
        (0.0, 1.0, 0.0)    # 3 undefined = green
    ])

    with torch.no_grad():
        for i, (_, _, img, msk) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            img = img.to(device)
            msk = msk.to(device)

            # BYOL representation
            z, _ = model.forward_byol(img)

            # Segmentation logits
            pred = model.forward_segmentation(img)  # shape: (B, n_classes, H, W)

            all_z.append(z.cpu())
            all_imgs.append(img.cpu())
            all_preds.append(pred.cpu())
            all_masks.append(msk.cpu())

    # PCA on BYOL features
    all_z = torch.cat(all_z)
    z_np = all_z.numpy()
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_np)

    plt.figure(figsize=(5, 5))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6)
    plt.title(f"{tag} Latent space projection (BYOL features)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualize a random sample
    all_imgs = torch.cat(all_imgs)
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    idx = random.randint(0, all_imgs.size(0) - 1)

    img = all_imgs[idx].permute(1, 2, 0).numpy()
    logits = all_preds[idx]  # (n_classes, H, W)
    mask = all_masks[idx].squeeze().numpy()  # (H, W)

    # Normalize image
    img = (img - img.min()) / (img.max() - img.min())

    # Predicted class map
    probs = torch.softmax(logits, dim=0).numpy()
    pred_classes = probs.argmax(axis=0)  # (H, W)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"{tag} Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_classes, cmap=cmap, vmin=0, vmax=3)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=3)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def train_epoch(model, train_loader, optimizer, byol_loss, segmentation_loss, lambda_ssl, lambda_seg, device):
    """
    Training loop for one epoch combining BYOL self-supervised learning and weakly supervised segmentation
    """
    model = model.to(device)

    model.train()
    epoch_loss = 0
    epoch_byol_loss = 0
    epoch_seg_loss = 0

    pbar = tqdm(train_loader, desc = 'Training')
    for img1, img2, img, msk in pbar:
        # Move data to device
        img1, img2 = img1.to(device), img2.to(device)
        img = img.to(device)
        msk = msk.long().to(device) 

        # Forward pass for both views
        z1, p1 = model.forward_byol(img1)
        z2, p2 = model.forward_byol(img2)
        pred = model.forward_segmentation(img)

        # Calculate losses
        loss_byol = byol_loss(z1, p2, z2, p1)
        loss_seg = segmentation_loss(pred, msk)
        loss = lambda_ssl * loss_byol + lambda_seg * loss_seg
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        epoch_loss += loss.item()
        epoch_byol_loss += loss_byol.item()
        epoch_seg_loss += loss_seg.item()

        pbar.set_postfix({
            'loss': loss.item(),
            'byol': loss_byol.item(),
            'seg': loss_seg.item()
        })

    # Calculate average losses
    avg_loss = epoch_loss / len(train_loader)
    avg_byol = epoch_byol_loss / len(train_loader)
    avg_seg = epoch_seg_loss / len(train_loader)

    return avg_loss, avg_byol, avg_seg

def evaluate(model, val_loader, byol_loss, segmentation_loss, lambda_ssl, lambda_seg, device):
    """
    Evaluation function for the model on validation data.
    Includes Accuracy, Precision, IoU and F1 for multiclass segmentation.
    """
    model = model.to(device)
    model.eval()
    
    val_loss, val_byol_loss, val_seg_loss = 0, 0, 0
    all_preds, all_masks = [], []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluating')
        for img1, img2, img, msk in pbar:
            # Move data to device
            img1, img2 = img1.to(device), img2.to(device)
            img, msk = img.to(device), msk.long().to(device)

            # Forward pass
            z1, p1 = model.forward_byol(img1)
            z2, p2 = model.forward_byol(img2)
            pred = model.forward_segmentation(img)

            # Losses
            loss_byol = byol_loss(z1, p2, z2, p1)
            loss_seg = segmentation_loss(pred, msk)
            loss = lambda_ssl * loss_byol + lambda_seg * loss_seg

            val_loss += loss.item()
            val_byol_loss += loss_byol.item()
            val_seg_loss += loss_seg.item()

            # Predictions for metrics
            pred_classes = torch.argmax(pred, dim=1)
            all_preds.append(pred_classes.cpu().numpy().ravel())
            all_masks.append(msk.cpu().numpy().ravel())

    # Concatenate predictions and ground truth
    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Metrics
    acc = accuracy_score(all_masks, all_preds)
    prec = precision_score(all_masks, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_masks, all_preds, average="macro", zero_division=0)

    # IoU per class
    ious = []
    for cls in np.unique(all_masks):
        intersection = np.logical_and(all_preds == cls, all_masks == cls).sum()
        union = np.logical_or(all_preds == cls, all_masks == cls).sum()
        ious.append(intersection / union if union > 0 else 0.0)
    mean_iou = np.mean(ious)

    # Averages
    avg_loss = val_loss / len(val_loader)
    avg_byol = val_byol_loss / len(val_loader)
    avg_seg = val_seg_loss / len(val_loader)

    metrics = {
        "loss": avg_loss,
        "byol": avg_byol,
        "seg": avg_seg,
        "accuracy": acc,
        "precision": prec,
        "f1": f1,
        "iou": mean_iou
    }

    return metrics


def train_model(model, train_loader, val_loader, optimizer, byol_loss, segmentation_loss,
                lambda_ssl, lambda_seg, device, epochs, early_stopping = None, save_path = None,
                visualize = True):
    """
    Full training loop with evaluation and optional early stopping
    """

    best_val_loss = float('inf')
    no_improve = 0
    
    train_history = {'loss': [], 'byol': [], 'seg': []}
    val_history = {'loss': [], 'byol': [], 'seg': [], 'accuracy': [], 'precision': [], 'f1': [],  'iou': []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Training
        train_loss, train_byol, train_seg = train_epoch(
            model, train_loader, optimizer, byol_loss, segmentation_loss,
            lambda_ssl, lambda_seg, device
        )
        train_history['loss'].append(train_loss)
        train_history['byol'].append(train_byol)
        train_history['seg'].append(train_seg)

        if visualize and epoch % 10 == 0:
            visualize_pca_and_segmentation(model, train_loader, device, tag="Train", max_batches=5)

        # Evaluation
        metrics = evaluate(
            model, val_loader, byol_loss, segmentation_loss,
            lambda_ssl, lambda_seg, device
        )
    
        val_history['loss'].append(metrics['loss'])
        val_history['byol'].append(metrics['byol'])
        val_history['seg'].append(metrics['seg'])
        val_history['accuracy'].append(metrics['accuracy'])
        val_history['precision'].append(metrics['precision'])
        val_history['f1'].append(metrics['f1'])
        val_history['iou'].append(metrics['iou'])

        if visualize and epoch % 10 == 0:
            visualize_pca_and_segmentation(model, val_loader, device, tag="Val", max_batches=5)

        print(f"Train Loss: {train_loss:.4f} (BYOL: {train_byol:.4f}, Seg: {train_seg:.4f})")
        print(f"Val Loss: {metrics['loss']:.4f} (BYOL: {metrics['byol']:.4f}, Seg: {metrics['seg']:.4f})")

        # Early stopping check
        if save_path:
            if metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']
                best_epoch = epoch
                no_improve = 0

                # Save best model
                torch.save(model.state_dict(), save_path)
                print(f"Model saved on epoch {best_epoch}")
            else:
                no_improve += 1
                if early_stopping and no_improve >= early_stopping:
                    print(f"Early stopping after {no_improve} epochs without improvement. Last model saved: epoch {best_epoch}.")
                    break
    
    return train_history, val_history

def calculate_metrics_multiclass(preds, target, num_classes, reduction = "macro"):
    preds_soft = torch.softmax(preds, dim=1)
    preds_label = torch.argmax(preds_soft, dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(preds_label, target, mode = 'multiclass', num_classes=num_classes)

    f1score = smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction)
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction)
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction=reduction)
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction=reduction)
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction=reduction)

    return {'f1score': f1score.cpu().numpy(),
            'iou': iou.cpu().numpy(),
            'precision': precision.cpu().numpy(),
            'recall': recall.cpu().numpy(),
            'accuracy': accuracy.cpu().numpy()}
