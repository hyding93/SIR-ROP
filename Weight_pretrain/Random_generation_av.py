import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def morphological_skeleton(binary: np.ndarray) -> np.ndarray:
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def find_branch_points(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    branch_points = ((skeleton > 0) & (neighbor_count >= 3)).astype(np.uint8)
    return branch_points


def colorize_vessel_mask(mask: np.ndarray, red_ratio_range=(0.30, 0.50), min_seg_len=8):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    skeleton = morphological_skeleton(binary)
    branch_points = find_branch_points(skeleton)
    skeleton_segments = skeleton.copy()
    skeleton_segments[branch_points > 0] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton_segments, connectivity=8
    )

    if num_labels <= 1:
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color = [0, 0, 255] if np.random.rand() < 0.4 else [255, 0, 0]
        colored[binary > 0] = color
        return colored

    segments = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_seg_len:
            segments.append({'label': i, 'area': area})

    if not segments:
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored[binary > 0] = [255, 0, 0]
        return colored

    total_skel_area = sum(s['area'] for s in segments)
    target_red = int(total_skel_area * np.random.uniform(*red_ratio_range))

    segments.sort(key=lambda x: x['area'], reverse=True)
    red_labels = set()
    collected = 0
    for s in segments:
        if abs(collected + s['area'] - target_red) <= abs(collected - target_red):
            red_labels.add(s['label'])
            collected += s['area']

    skel_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for s in segments:
        color = [0, 0, 255] if s['label'] in red_labels else [255, 0, 0]
        skel_color[labels == s['label']] = color

    red_mask = (skel_color[:, :, 2] == 255).astype(np.uint8)
    blue_mask = (skel_color[:, :, 0] == 255).astype(np.uint8)

    if red_mask.sum() == 0:
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored[binary > 0] = [255, 0, 0]
        return colored
    if blue_mask.sum() == 0:
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored[binary > 0] = [0, 0, 255]
        return colored

    dist_red = cv2.distanceTransform(1 - red_mask, cv2.DIST_L2, 5)
    dist_blue = cv2.distanceTransform(1 - blue_mask, cv2.DIST_L2, 5)

    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vessel_ys, vessel_xs = np.where(binary > 0)

    for y, x in zip(vessel_ys, vessel_xs):
        if dist_red[y, x] <= dist_blue[y, x]:
            colored[y, x] = [0, 0, 255]
        else:
            colored[y, x] = [255, 0, 0]

    return colored


def batch_process(input_dir: str, output_dir: str,
                  red_ratio_range=(0.30, 0.50),
                  extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir()
             if f.suffix.lower() in extensions and f.is_file()]

    if not files:
        print(f"In {input_dir} no pictures!")
        return

    print(f"A total of {len(files)} pictures were found")

    for file_path in tqdm(files, desc="being processed"):
        mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f" warning ：Unable to read {file_path.name}， skip!")
            continue

        colored = colorize_vessel_mask(mask, red_ratio_range=red_ratio_range)
        cv2.imwrite(str(output_dir / file_path.name), colored)

    print(f"Fulfille the all , save to : {output_dir}")

if __name__ == "__main__":
    # ==================== 使用示例 ====================
    INPUT_DIR = r"./training/av/"
    OUTPUT_DIR = r"./training/av_mask/"

    # red proportion：30% ~ 50%
    batch_process(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        red_ratio_range=(0.30, 0.50)
    )
