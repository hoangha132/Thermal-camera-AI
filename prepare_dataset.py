import os
import zipfile
import shutil
import random
from glob import glob
from tqdm import tqdm
import albumentations as A
import cv2
import yaml

# --------------------
# CONFIG
# --------------------
DATASETS = {
    "people": {"zip": "people.zip", "class_id": 0},
    "car": {"zip": "car.zip", "class_id": 1},
    "drone": {"zip": "drone.zip", "class_id": 2},
}

OUTPUT_DIR = "merged_dataset"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
MAX_PER_CLASS = 1000   # cap number of images per class

AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

# --------------------
# UTILS
# --------------------
def unzip_dataset(zip_path, out_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

def read_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r") as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append(parts)
        elif len(parts) > 5:
            bbox = seg_to_bbox(line)
            labels.append(list(map(str, bbox)))
    return labels

def write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for l in labels:
            f.write(" ".join(map(str, l)) + "\n")

def relabel(label_path, new_class_id):
    labels = read_labels(label_path)
    new_labels = []
    for l in labels:
        l[0] = str(new_class_id)
        new_labels.append(l)
    write_labels(label_path, new_labels)

def seg_to_bbox(line):
    parts = list(map(float, line.strip().split()))
    cls = int(parts[0])
    coords = parts[1:]

    xs = coords[0::2]
    ys = coords[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    return [cls, x_center, y_center, w, h]

def augment_image(img_path, label_path, out_img_path, out_label_path):
    image = cv2.imread(img_path)
    if image is None:
        return
    h, w = image.shape[:2]
    labels = read_labels(label_path)

    bboxes = []
    class_ids = []
    for l in labels:
        cls, x, y, bw, bh = map(float, l)
        class_ids.append(int(cls))
        bboxes.append([x, y, bw, bh])

    if len(bboxes) == 0:
        return

    transformed = AUG(image=image, bboxes=bboxes, class_labels=class_ids)
    aug_img = transformed['image']
    aug_bboxes = transformed['bboxes']
    aug_classes = transformed['class_labels']

    cv2.imwrite(out_img_path, aug_img)

    new_labels = []
    for cls, (x, y, bw, bh) in zip(aug_classes, aug_bboxes):
        new_labels.append([cls, x, y, bw, bh])
    write_labels(out_label_path, new_labels)

def split_dataset(data, out_dir, class_id):
    random.shuffle(data)

    # split by percentages
    n_total = len(data)
    n_train = int(n_total * SPLITS["train"])
    n_val = int(n_total * SPLITS["val"])

    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train+n_val],
        "test": data[n_train+n_val:]
    }

    for split, files in splits.items():
        for img_path, label_path in tqdm(files, desc=f"{split}-{class_id}"):
            out_img = os.path.join(out_dir, "images", split, os.path.basename(img_path))
            out_label = os.path.join(out_dir, "labels", split, os.path.basename(label_path))

            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            os.makedirs(os.path.dirname(out_label), exist_ok=True)

            shutil.copy(img_path, out_img)
            shutil.copy(label_path, out_label)

            # Augmentation
            out_img_aug = out_img.replace(".jpg", "_aug.jpg")
            out_label_aug = out_label.replace(".txt", "_aug.txt")
            augment_image(out_img, out_label, out_img_aug, out_label_aug)

def make_yaml(out_dir):
    data = {
        "train": f"{out_dir}/images/train",
        "val": f"{out_dir}/images/val",
        "test": f"{out_dir}/images/test",
        "nc": 3,
        "names": ["people", "car", "drone"]
    }
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False)

# --------------------
# MAIN
# --------------------
def main():
    temp_dir = "temp_datasets"
    os.makedirs(temp_dir, exist_ok=True)

    datasets_all = {}

    # 1. Giải nén và chuẩn hóa label
    for name, info in DATASETS.items():
        zip_path = info["zip"]
        out_dir = os.path.join(temp_dir, name)
        unzip_dataset(zip_path, out_dir)

        images = glob(os.path.join(out_dir, "**", "*.jpg"), recursive=True)
        labels = [img.replace("images", "labels").replace(".jpg", ".txt") for img in images]

        paired = [(img, lbl) for img, lbl in zip(images, labels) if os.path.exists(lbl)]

        # Relabel
        for lbl in labels:
            if os.path.exists(lbl):
                relabel(lbl, info["class_id"])

        # Limit to MAX_PER_CLASS
        if len(paired) > MAX_PER_CLASS:
            paired = random.sample(paired, MAX_PER_CLASS)

        datasets_all[info["class_id"]] = paired

    # 2. Split & copy
    for class_id, data in datasets_all.items():
        split_dataset(data, OUTPUT_DIR, class_id)

    # 3. Make yaml
    make_yaml(OUTPUT_DIR)

    # 4. Zip final dataset
    shutil.make_archive("dataset_ready", 'zip', OUTPUT_DIR)
    print("✅ Dataset chuẩn bị xong (capped to 1000/class): dataset_ready.zip")

if __name__ == "__main__":
    main()
