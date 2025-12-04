import os, zipfile, shutil, random, cv2, numpy as np

# =============== SETTINGS ===============
FINAL_CLASSES = [
    "human", "car", "tank", "bird", "drone", "helicopter", "missile", "plane"
]

ZIP_NAMES = {
    "car": "car_rgb.zip",
    "tank": "tank_rgb.zip",
    "people": "people_rgb.zip",
    "air": "air_rgb.zip",
}

OUT_DIR = "merged_thermal_dataset"
TEMP_DIR = "temp_merge"
FINAL_ZIP = "merged_thermal_dataset.zip"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
THERMAL_MODE = "whitehot"
# ========================================



# ==================================================================
#   ★★★★★ REALISTIC RGB → WHITE-HOT THERMAL IMAGE CONVERTER ★★★★★
# ==================================================================
def rgb_to_realistic_thermal(image):
    """
    Convert RGB to realistic synthetic thermal — white-hot.

    This simulates heat radiance:
      - Bright objects appear hot
      - Skin areas boosted (people stand out)
      - Texture smoothing (thermal cameras blur textures)
      - Final equalized white-hot tone
    """

    # 1) Start with luminance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    base_heat = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 2) Detect human skin regions → they should always appear HOT
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)

    # 3) Texture → thermal smooth (radiation diffuses)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    thermal_smooth = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    # 4) Combine simulated radiance channels
    combined = (
        0.55 * base_heat + 
        1.25 * skin_mask + 
        0.40 * thermal_smooth
    )

    combined = np.clip(combined, 0, 255).astype(np.uint8)

    # 5) White-hot contrast boost
    thermal = cv2.equalizeHist(combined)

    # 6) Convert to 3-channel
    thermal_rgb = cv2.merge([thermal, thermal, thermal])

    return thermal_rgb



# ==================================================================
# --------------------------- UNZIP ALL -----------------------------
# ==================================================================
def unzip_all():
    """Extract all input datasets."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    for key, zipname in ZIP_NAMES.items():
        if not os.path.exists(zipname):
            raise FileNotFoundError(f"❌ Missing required zip: {zipname}")

        dst = os.path.join(TEMP_DIR, key)
        os.makedirs(dst, exist_ok=True)

        print(f"📦 Unzipping {zipname} ...")
        with zipfile.ZipFile(zipname, "r") as z:
            z.extractall(dst)

    print("✅ All datasets extracted.")



# ==================================================================
# ------------------------- YOLO SPLITTER --------------------------
# ==================================================================
def ensure_split(base_dir):
    """Ensure YOLO format with guaranteed train/val/test splits."""
    if os.path.exists(os.path.join(base_dir, "train")):
        return

    print(f"🛠️ Splitting dataset in {base_dir} ...")

    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")

    all_imgs = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    random.shuffle(all_imgs)
    n = len(all_imgs)

    if n < 3:
        raise ValueError(f"❌ Not enough images in {base_dir}. Need ≥ 3, found {n}")

    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    n_test = n - n_train - n_val

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train + n_val],
        "test": all_imgs[n_train + n_val:]
    }

    for split, files in splits.items():
        split_img = os.path.join(base_dir, split, "images")
        split_lbl = os.path.join(base_dir, split, "labels")
        os.makedirs(split_img, exist_ok=True)
        os.makedirs(split_lbl, exist_ok=True)

        for f in files:
            shutil.move(os.path.join(img_dir, f), os.path.join(split_img, f))

            lbl_name = os.path.splitext(f)[0] + ".txt"
            if os.path.exists(os.path.join(lbl_dir, lbl_name)):
                shutil.move(os.path.join(lbl_dir, lbl_name), os.path.join(split_lbl, lbl_name))

    shutil.rmtree(img_dir)
    shutil.rmtree(lbl_dir)
    print(f"   → {n_train} train, {n_val} val, {n_test} test.")



# ==================================================================
# ----------- COPY + THERMAL CONVERT + LABEL OFFSET ----------------
# ==================================================================
def copy_and_convert(src_dir, dst_dir, class_offset=0):
    """Copy data from source, apply thermal conversion, and adjust class ids."""
    for split in ["train", "val", "test"]:
        src_img = os.path.join(src_dir, split, "images")
        src_lbl = os.path.join(src_dir, split, "labels")
        dst_img = os.path.join(dst_dir, split, "images")
        dst_lbl = os.path.join(dst_dir, split, "labels")

        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        if not os.path.exists(src_img):
            continue

        img_files = [
            f for f in os.listdir(src_img)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        for f in img_files:

            # Convert RGB → thermal
            rgb = cv2.imread(os.path.join(src_img, f))
            thermal = rgb_to_realistic_thermal(rgb)
            cv2.imwrite(os.path.join(dst_img, f), thermal)

            # Label rewriting
            in_lbl = os.path.join(src_lbl, os.path.splitext(f)[0] + ".txt")
            out_lbl = os.path.join(dst_lbl, os.path.splitext(f)[0] + ".txt")

            if os.path.exists(in_lbl):
                with open(in_lbl, "r") as fin, open(out_lbl, "w") as fout:
                    for line in fin:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            parts[0] = str(int(parts[0]) + class_offset)
                            fout.write(" ".join(parts) + "\n")



# ==================================================================
# ---------------------------- YAML --------------------------------
# ==================================================================
def make_data_yaml(out_dir):
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {len(FINAL_CLASSES)}\n")
        f.write("names:\n")
        for i, n in enumerate(FINAL_CLASSES):
            f.write(f"  {i}: {n}\n")

    print(f"🧾 data.yaml created at {yaml_path}")



# ==================================================================
# ----------------------------- ZIP --------------------------------
# ==================================================================
def zip_output(out_dir, zip_name):
    print("📦 Creating final zip...")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for file in files:
                fp = os.path.join(root, file)
                z.write(fp, os.path.relpath(fp, out_dir))
    print(f"✅ Final dataset zipped as {zip_name}")



# ==================================================================
# ----------------------------- MAIN -------------------------------
# ==================================================================
def main():
    unzip_all()

    # Ensure each dataset has train/val/test structure
    for key in ZIP_NAMES:
        ensure_split(os.path.join(TEMP_DIR, key))

    # Prepare output base
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

    # Merge with correct class offset:
    print("🔥 Converting & merging datasets into thermal style...")

    copy_and_convert(os.path.join(TEMP_DIR, "people"), OUT_DIR, class_offset=0)
    copy_and_convert(os.path.join(TEMP_DIR, "car"), OUT_DIR, class_offset=1)
    copy_and_convert(os.path.join(TEMP_DIR, "tank"), OUT_DIR, class_offset=2)
    copy_and_convert(os.path.join(TEMP_DIR, "air"), OUT_DIR, class_offset=3)

    print("✅ Merge complete.")

    make_data_yaml(OUT_DIR)
    zip_output(OUT_DIR, FINAL_ZIP)

    print("\n🎯 DONE — Thermal dataset ready for YOLO training!")


if __name__ == "__main__":
    random.seed(42)
    main()
