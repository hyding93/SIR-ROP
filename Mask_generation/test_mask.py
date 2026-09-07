import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

def main():
    MODEL_WEIGHTS = "./weight/best_unet_model.pth"
    NEW_DATASET_ROOT = "./data/"
    OUTPUT_ROOT = "./mask_out/"
    IMG_SIZE = (512, 512)
    THRESHOLD = 0.5
    print("Building model and loading the best weights...")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lesion_model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Weight file not found: {MODEL_WEIGHTS}")
        return

    lesion_model.load_state_dict(
        torch.load(MODEL_WEIGHTS, map_location=device)
    )
    lesion_model.to(device)
    lesion_model.eval()

    print("Model loaded successfully!\n")

    image_paths = []
    for root, dirs, files in os.walk(NEW_DATASET_ROOT):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image_paths.append(os.path.join(root, file))

    print(
        f"Found {len(image_paths)} images in the target dataset. "
        "Starting batch processing...\n"
    )

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Prediction progress"):
            rel_path = os.path.relpath(img_path, NEW_DATASET_ROOT)
            rel_path_no_ext = os.path.splitext(rel_path)[0]
            out_rel_path = rel_path_no_ext + ".png"

            save_path = os.path.join(OUTPUT_ROOT, out_rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            orig_img = cv2.imread(img_path)
            if orig_img is None:
                continue

            orig_h, orig_w = orig_img.shape[:2]

            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, IMG_SIZE)

            img_tensor = img_resized.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (2, 0, 1))
            img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(device)

            pred = lesion_model(img_tensor)
            pred_prob = torch.sigmoid(pred).squeeze().cpu().numpy()

            pred_mask = (pred_prob > THRESHOLD).astype(np.uint8) * 255
            pred_mask_orig_size = cv2.resize(
                pred_mask,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(save_path, pred_mask_orig_size)
    print(f"\nProcessing completed! Output saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()