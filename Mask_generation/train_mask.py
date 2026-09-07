import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp

class FundusDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.image_names = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        mask_name = os.path.splitext(img_name)[0] + ".jpg"
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Unable to read the image; please check the path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read the tag; please check the path: {mask_path}")

        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)


def calculate_dice(preds, targets, threshold=0.5, epsilon=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    dice = (2.0 * intersection + epsilon) / (preds_flat.sum() + targets_flat.sum() + epsilon)
    return dice.item()


def save_visualization(image_tensor, true_mask_tensor, pred_tensor, epoch, save_dir):
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    true_mask_np = true_mask_tensor.cpu().numpy().squeeze()

    pred_prob = torch.sigmoid(pred_tensor).detach().cpu().numpy().squeeze()
    pred_mask_np = (pred_prob > 0.5).astype(np.float32)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image (Val Set)")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(true_mask_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Predicted Mask (Epoch {epoch})")
    plt.imshow(pred_mask_np, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch}_vis.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    TRAIN_IMG_DIR = "./img_square/"
    TRAIN_MASK_DIR = "./mask_square/"
    OUTPUT_DIR = "./weight/"
    VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

    os.makedirs(VIS_DIR, exist_ok=True)

    BATCH_SIZE = 8
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-4

    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(TRAIN_MASK_DIR):
        print("Error: Dataset folder not found. Please check the path.")
        return

    full_dataset = FundusDataset(images_dir=TRAIN_IMG_DIR, masks_dir=TRAIN_MASK_DIR)

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Dataset successfully loaded; a total of{len(full_dataset)} group.")
    print(f"Training set: {train_size} group | Validation set: {val_size} group")

    lesion_model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).cuda()

    optimizer = torch.optim.Adam(lesion_model.parameters(), lr=LEARNING_RATE)
    loss_fn = smp.losses.DiceLoss(mode='binary')

    print("================ Training ================")
    best_val_dice = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_unet_model.pth")
    last_model_path = os.path.join(OUTPUT_DIR, "last_unet_model.pth")

    for epoch in range(1, EPOCHS + 1):
        lesion_model.train()
        train_loss = 0
        train_dice = 0

        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()

            predictions = lesion_model(images)
            loss = loss_fn(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += calculate_dice(predictions, masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        lesion_model.eval()
        val_loss = 0
        val_dice = 0
        vis_image, vis_mask, vis_pred = None, None, None

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.cuda()
                masks = masks.cuda()

                predictions = lesion_model(images)
                loss = loss_fn(predictions, masks)

                val_loss += loss.item()
                val_dice += calculate_dice(predictions, masks)

                if batch_idx == 0:
                    vis_image = images[0]
                    vis_mask = masks[0]
                    vis_pred = predictions[0]

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"Epoch [{epoch:03d}/{EPOCHS}] "
              f"| Train Loss: {avg_train_loss:.4f} - Dice: {avg_train_dice:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} - Dice: {avg_val_dice:.4f}")

        if vis_image is not None:
            save_visualization(vis_image, vis_mask, vis_pred, epoch, VIS_DIR)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(lesion_model.state_dict(), best_model_path)
            print(f"***Better model discovered! Current best: Val Dice: {best_val_dice:.4f}, the best weights have been saved!")

    torch.save(lesion_model.state_dict(), last_model_path)
    print(f"\n================ Training fully completed ================")
    print(f"***The best model has been saved to：{best_model_path} (Val Dice: {best_val_dice:.4f})")
    print(f"***The final round of the model has been saved to:{last_model_path}")


if __name__ == "__main__":
    main()