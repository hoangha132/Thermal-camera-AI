import os, zipfile, shutil, random, cv2, numpy as np

# =============== SETTINGS ===============
FINAL_CLASSES = [
    "human", "car", "tank", "bird", "drone", "helicopter", "missile", "plane"
]

# Enhanced thermal characteristics with challenge scenarios
THERMAL_PROFILES = {
    "human": {"base_temp": 0.8, "hot_spots": 0.9, "consistency": 0.9, "heat_strength": 0.8},
    "car": {"base_temp": 0.7, "hot_spots": 0.8, "consistency": 0.8, "heat_strength": 0.7},
    "tank": {"base_temp": 0.6, "hot_spots": 0.7, "consistency": 0.7, "heat_strength": 0.6},
    "bird": {"base_temp": 0.9, "hot_spots": 0.6, "consistency": 0.5, "heat_strength": 0.4},
    "drone": {"base_temp": 0.5, "hot_spots": 0.7, "consistency": 0.6, "heat_strength": 0.5},
    "helicopter": {"base_temp": 0.7, "hot_spots": 0.9, "consistency": 0.8, "heat_strength": 0.7},
    "missile": {"base_temp": 0.9, "hot_spots": 0.95, "consistency": 0.9, "heat_strength": 0.9},
    "plane": {"base_temp": 0.6, "hot_spots": 0.8, "consistency": 0.7, "heat_strength": 0.6}
}

# Challenge scenarios for robust training
CHALLENGE_SCENARIOS = {
    "clear": {"probability": 0.5, "contrast_range": (0.7, 1.0), "noise_level": 0.1},
    "foggy": {"probability": 0.2, "contrast_range": (0.3, 0.6), "noise_level": 0.3},
    "rainy": {"probability": 0.15, "contrast_range": (0.4, 0.7), "noise_level": 0.4},
    "extreme_low_contrast": {"probability": 0.15, "contrast_range": (0.1, 0.4), "noise_level": 0.5}
}

# Mapping from source datasets to final class indices
DATASET_CLASS_MAPPING = {
    "people": {0: 0},  # people dataset: class 0 -> human (0)
    "car": {0: 1},     # car dataset: class 0 -> car (1)
    "tank": {0: 2},    # tank dataset: class 0 -> tank (2)
    "air": {           # air dataset: Roboflow 5-class mapping
        0: 3,  # Bird -> bird (3)
        1: 4,  # Drone -> drone (4)
        2: 5,  # Helicopter -> helicopter (5)
        3: 6,  # Missile -> missile (6)
        4: 7   # Plane -> plane (7)
    }
}

# Primary class for thermal conversion for each dataset
DATASET_PRIMARY_CLASS = {
    "people": "human",
    "car": "car", 
    "tank": "tank",
    "air": "drone"
}

ZIP_NAMES = {
    "car": "car_rgb.zip",
    "tank": "tank_rgb.zip", 
    "people": "people_rgb.zip",
    "air": "air_rgb.zip",
}

OUT_DIR = "improved_thermal_dataset"
TEMP_DIR = "temp_merge"
FINAL_ZIP = "improved_thermal_dataset.zip"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
# ========================================

# ==================================================================
# ★★★★★ IMPROVED MILITARY-STYLE THERMAL CONVERSION ★★★★★
# ==================================================================
def rgb_to_military_thermal(image, class_name=None, challenge_scenario="clear"):
    """
    Military-style thermal: Bright objects on dark background
    with adaptive challenge scenarios for robust training.
    """
    
    # Get scenario parameters
    scenario = CHALLENGE_SCENARIOS[challenge_scenario]
    contrast_scale = random.uniform(*scenario["contrast_range"])
    noise_level = scenario["noise_level"]
    
    # 1) Start with luminance for base heat
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 2) Get thermal profile for this class
    if class_name and class_name in THERMAL_PROFILES:
        profile = THERMAL_PROFILES[class_name]
        base_temp = profile["base_temp"]
        hot_spot_strength = profile["hot_spots"]
        heat_strength = profile["heat_strength"]
    else:
        # Default thermal profile
        base_temp = 0.7
        hot_spot_strength = 0.8
        heat_strength = 0.7
    
    # 3) Base thermal mapping - emphasize object structure
    base_heat = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    base_heat = base_heat.astype(np.float32)
    
    # 4) Enhanced heat source simulation with class-specific characteristics
    # Find bright areas and amplify them as heat sources
    _, bright_mask = cv2.threshold(base_heat, 160, 255, cv2.THRESH_BINARY)
    bright_mask = bright_mask.astype(np.float32) / 255.0
    
    # 5) Create artificial hot spots based on object structure
    edges = cv2.Canny(image.astype(np.uint8), 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edge_dilated = cv2.dilate(edges, kernel, iterations=1)
    edge_heat = edge_dilated.astype(np.float32) * hot_spot_strength * 0.5
    
    # 6) Engine/heat source simulation for vehicles
    engine_heat = np.zeros_like(base_heat)
    if class_name in ["car", "tank", "helicopter", "plane", "drone"]:
        # Simulate engine heat as bright spots
        height, width = base_heat.shape
        engine_center_x = random.randint(width//4, 3*width//4)
        engine_center_y = random.randint(height//4, 3*height//4)
        
        y, x = np.ogrid[:height, :width]
        mask = ((x - engine_center_x)**2 + (y - engine_center_y)**2) <= 100
        engine_heat[mask] = random.uniform(0.6, 0.9) * 255
    
    # 7) Skin detection for humans (always hot)
    skin_heat = np.zeros_like(base_heat)
    if class_name == "human":
        try:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
            skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
            skin_heat = skin_mask.astype(np.float32) * heat_strength
        except:
            pass
    
    # 8) Combine all thermal components
    combined = (
        0.4 * base_heat +
        0.3 * bright_mask * 255 * heat_strength +
        0.1 * edge_heat +
        0.1 * engine_heat +
        0.1 * skin_heat
    )
    
    # 9) Apply challenge scenario effects
    # Reduce contrast for challenging scenarios
    combined = combined * contrast_scale
    
    # Add thermal noise
    noise = np.random.normal(0, noise_level * 25, combined.shape)
    combined = np.clip(combined + noise, 0, 255).astype(np.uint8)
    
    # 10) CRITICAL: Invert for military thermal (bright objects on dark background)
    thermal = 255 - combined
    
    # 11) Final enhancements for military thermal
    # Aggressive contrast stretching for the inverted image
    min_val = np.percentile(thermal, 10)
    max_val = np.percentile(thermal, 90)
    thermal = cv2.normalize(thermal, None, min_val, max_val, cv2.NORM_MINMAX)
    
    # Apply gamma correction to enhance bright objects
    gamma = 0.7  # Brighten the bright objects
    thermal = np.power(thermal / 255.0, gamma) * 255
    thermal = np.clip(thermal, 0, 255).astype(np.uint8)
    
    # Final histogram equalization to maximize contrast
    thermal = cv2.equalizeHist(thermal)
    
    # Convert to 3-channel
    thermal_rgb = cv2.merge([thermal, thermal, thermal])
    
    return thermal_rgb

def apply_thermal_challenges(image, class_name):
    """
    Apply random challenge scenarios to make training more robust.
    """
    # Randomly select challenge scenario based on probabilities
    rand_val = random.random()
    cumulative_prob = 0
    selected_scenario = "clear"
    
    for scenario, params in CHALLENGE_SCENARIOS.items():
        cumulative_prob += params["probability"]
        if rand_val <= cumulative_prob:
            selected_scenario = scenario
            break
    
    return rgb_to_military_thermal(image, class_name, selected_scenario)

# ==================================================================
# ★★★★★ MULTI-CLASS LABEL PROCESSING ★★★★★
# ==================================================================
def process_labels(src_lbl_file, dst_lbl_file, class_mapping):
    """
    Process labels by mapping source class IDs to final class IDs.
    """
    if not os.path.exists(src_lbl_file):
        return
        
    with open(src_lbl_file, "r") as fin, open(dst_lbl_file, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) == 5:
                src_class_id = int(parts[0])
                # Map source class ID to final class ID using the mapping
                if src_class_id in class_mapping:
                    final_class_id = class_mapping[src_class_id]
                    parts[0] = str(final_class_id)
                    fout.write(" ".join(parts) + "\n")
                else:
                    print(f"⚠️ Warning: Unknown class ID {src_class_id} in {src_lbl_file}")

# ==================================================================
# ★★★★★ CREATE CONSISTENT SPLITS FOR ALL DATASETS ★★★★★
# ==================================================================
def create_consistent_splits(dataset_path):
    """
    Ensure every dataset has train/val/test splits with 80/10/10 ratio.
    """
    # Check if splits already exist in standard format
    if (os.path.exists(os.path.join(dataset_path, "train")) and 
        os.path.exists(os.path.join(dataset_path, "val")) and 
        os.path.exists(os.path.join(dataset_path, "test"))):
        print(f"📁 Using existing standard splits for {os.path.basename(dataset_path)}")
        return
    
    print(f"🛠️ Creating consistent splits for {os.path.basename(dataset_path)}...")
    
    # Create temporary working directory
    temp_working_dir = os.path.join(dataset_path, "temp_split")
    os.makedirs(temp_working_dir, exist_ok=True)
    
    # Collect all images and labels
    all_images = []
    all_labels = {}
    
    # Search through all possible locations
    search_locations = [
        ("train", "train"),
        ("val", "val"), 
        ("valid", "val"),
        ("test", "test"),
        ("", "root")  # root images/labels directories
    ]
    
    for dir_name, split_type in search_locations:
        if dir_name:  # For split directories
            img_dir = os.path.join(dataset_path, dir_name, "images")
            lbl_dir = os.path.join(dataset_path, dir_name, "labels")
        else:  # For root directories
            img_dir = os.path.join(dataset_path, "images")
            lbl_dir = os.path.join(dataset_path, "labels")
        
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) 
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            
            for img in images:
                img_path = os.path.join(img_dir, img)
                lbl_name = os.path.splitext(img)[0] + ".txt"
                lbl_path = os.path.join(lbl_dir, lbl_name) if os.path.exists(lbl_dir) else None
                
                # Copy to temporary working directory
                temp_img_path = os.path.join(temp_working_dir, img)
                if not os.path.exists(temp_img_path):
                    shutil.copy2(img_path, temp_img_path)
                
                if lbl_path and os.path.exists(lbl_path):
                    temp_lbl_path = os.path.join(temp_working_dir, lbl_name)
                    if not os.path.exists(temp_lbl_path):
                        shutil.copy2(lbl_path, temp_lbl_path)
                
                all_images.append(img)
                if lbl_path and os.path.exists(lbl_path):
                    all_labels[img] = lbl_name
    
    if not all_images:
        print(f"⚠️ No images found in {dataset_path}")
        shutil.rmtree(temp_working_dir)
        return
    
    # Remove duplicates and shuffle
    all_images = list(set(all_images))
    random.shuffle(all_images)
    n = len(all_images)
    
    if n < 3:
        print(f"⚠️ Not enough images in {dataset_path} for splitting. Need ≥ 3, found {n}")
        shutil.rmtree(temp_working_dir)
        return
    
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    n_test = n - n_train - n_val
    
    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:]
    }
    
    # Create new split directories
    for split_name, image_list in splits.items():
        split_img_dir = os.path.join(dataset_path, split_name, "images")
        split_lbl_dir = os.path.join(dataset_path, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        
        for img_name in image_list:
            # Copy image from temp directory to new split directory
            src_img_path = os.path.join(temp_working_dir, img_name)
            dst_img_path = os.path.join(split_img_dir, img_name)
            
            if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            
            # Copy corresponding label if it exists
            if img_name in all_labels:
                lbl_name = all_labels[img_name]
                src_lbl_path = os.path.join(temp_working_dir, lbl_name)
                dst_lbl_path = os.path.join(split_lbl_dir, lbl_name)
                
                if os.path.exists(src_lbl_path) and not os.path.exists(dst_lbl_path):
                    shutil.copy2(src_lbl_path, dst_lbl_path)
    
    # Clean up old directories and temp directory
    shutil.rmtree(temp_working_dir)
    
    # Remove old split directories if they exist (but keep the new ones)
    old_dirs = ["images", "labels", "valid"]
    for old_dir in old_dirs:
        old_path = os.path.join(dataset_path, old_dir)
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
    
    print(f"   → {n_train} train, {n_val} val, {n_test} test images")

# ==================================================================
# ★★★★★ CLASS-AWARE THERMAL CONVERSION WITH CHALLENGES ★★★★★
# ==================================================================
def class_aware_thermal_conversion(src_dir, dst_dir, dataset_name):
    """Apply thermal conversion with proper class mapping and challenge scenarios."""
    
    class_mapping = DATASET_CLASS_MAPPING[dataset_name]
    primary_class = DATASET_PRIMARY_CLASS[dataset_name]
    
    for split_name in ["train", "val", "test"]:
        src_img_dir = os.path.join(src_dir, split_name, "images")
        src_lbl_dir = os.path.join(src_dir, split_name, "labels")
        dst_img_dir = os.path.join(dst_dir, split_name, "images")
        dst_lbl_dir = os.path.join(dst_dir, split_name, "labels")

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        if not os.path.exists(src_img_dir):
            print(f"⚠️ Skip {split_name} for {dataset_name} - no images found")
            continue

        img_files = [
            f for f in os.listdir(src_img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        print(f"🔄 Processing {dataset_name} {split_name}: {len(img_files)} images")

        for f in img_files:
            # Convert RGB → thermal with class awareness and challenges
            rgb = cv2.imread(os.path.join(src_img_dir, f))
            if rgb is None:
                print(f"⚠️ Could not read image: {f}")
                continue
            
            # Apply thermal conversion with random challenges for training
            if split_name == "train":
                # For training: apply random challenges to improve robustness
                thermal = apply_thermal_challenges(rgb, primary_class)
            else:
                # For val/test: use clear conditions for consistent evaluation
                thermal = rgb_to_military_thermal(rgb, primary_class, "clear")
                
            cv2.imwrite(os.path.join(dst_img_dir, f), thermal)

            # Label processing with class mapping
            in_lbl = os.path.join(src_lbl_dir, os.path.splitext(f)[0] + ".txt")
            out_lbl = os.path.join(dst_lbl_dir, os.path.splitext(f)[0] + ".txt")

            if os.path.exists(in_lbl):
                process_labels(in_lbl, out_lbl, class_mapping)

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
    # Clean and setup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        
    os.makedirs(TEMP_DIR, exist_ok=True)
    # Create output directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

    # Unzip datasets
    unzip_all()

    # Create consistent splits for ALL datasets
    print("\n🛠️ Creating consistent train/val/test splits for all datasets...")
    for dataset_name in ZIP_NAMES:
        dataset_path = os.path.join(TEMP_DIR, dataset_name)
        create_consistent_splits(dataset_path)

    # Merge with improved thermal conversion and challenge scenarios
    print("\n🔥 Converting & merging with MILITARY-STYLE thermal simulation...")
    print("🎯 Challenge scenarios for robust training:")
    for scenario, params in CHALLENGE_SCENARIOS.items():
        print(f"   • {scenario}: {params['probability']*100}% (contrast: {params['contrast_range']})")

    # Process each dataset with correct class mapping
    for dataset_name in DATASET_CLASS_MAPPING:
        print(f"\n🔄 Processing {dataset_name} dataset...")
        class_aware_thermal_conversion(
            os.path.join(TEMP_DIR, dataset_name), 
            OUT_DIR, 
            dataset_name
        )

    print("\n✅ IMPROVED THERMAL DATASET READY!")
    print("📊 Dataset features:")
    print(f"   • Military-style: Bright objects on dark background")
    print(f"   • Challenge scenarios: {len(CHALLENGE_SCENARIOS)} types")
    print(f"   • Class-specific thermal signatures")
    print(f"   • Enhanced robustness for extreme conditions")

    # Create YAML and final zip
    make_data_yaml(OUT_DIR)
    zip_output(OUT_DIR, FINAL_ZIP)

    # Cleanup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    print("\n🎯 TRAINING RECOMMENDATIONS:")
    print("   • Use YOLOv10s with img_size=640")
    print("   • Train for 50+ epochs")
    print("   • Enable mosaic and mixup augmentation")
    print("   • Monitor performance on val set with extreme conditions")

if __name__ == "__main__":
    random.seed(42)
    main()