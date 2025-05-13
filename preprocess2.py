import os
import numpy as np
import cv2
import shutil
from skimage.util import random_noise
from sklearn.utils import resample
from tqdm import tqdm

# 🚀 Configuration
BASE_DIR = r"C:\Project\Dataset\dataset"
OUTPUT_DIR = r"C:\Project\Enhanced_Dataset\dataset"
CLASSES = ['no', 'sphere', 'vort']
TARGET_SIZE = (150, 150)
AUGMENTATION_FACTOR = 5  # 5x original dataset size

# 🚀 Quality Enhancements
def enhance_data_quality(img):
    """Apply quality enhancements matching paper's simulation parameters"""
    # ✅ 1. PSF Blur (Airy disk approximation)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5, sigmaY=0.5)
    
    # ✅ 2. Add noise matching LSST specifications
    img = random_noise(img, mode='gaussian', var=0.001)
    
    # ✅ 3. Histogram stretching (contrast enhancement)
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    return img

# 🚀 Data Augmentation
def augment_image(img):
    """Apply augmentations to simulate LSST variations"""
    
    # ✅ Random rotation (0-360 degrees)
    angle = np.random.choice([0, 90, 180, 270])
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    
    # ✅ Random translation (±10%)
    tx = np.random.randint(-int(cols * 0.1), int(cols * 0.1))  # 10% of width
    ty = np.random.randint(-int(rows * 0.1), int(rows * 0.1))  # 10% of height
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (cols, rows))
    
    # ✅ Random flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    
    return np.clip(img, 0, 1)

# 🚀 Process and Augment Data
def process_and_save(class_dir, output_dir):
    """Process and augment images for a class"""
    os.makedirs(output_dir, exist_ok=True)

    img_files = os.listdir(class_dir)

    for img_file in tqdm(img_files, desc=f"Processing {class_dir}", unit="file"):
        img_path = os.path.join(class_dir, img_file)
        
        # ✅ Ensure valid file format
        if not img_file.endswith('.npy'):
            continue

        # ✅ Load and preprocess
        img = np.load(img_path).squeeze()

        # ✅ Skip corrupted or blank images
        if img.max() - img.min() < 0.01:
            print(f"Skipping corrupted file: {img_path}")
            continue

        # ✅ Normalize image
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # ✅ Apply quality enhancement
        enhanced_img = enhance_data_quality(img)

        # ✅ Save original enhanced version
        np.save(os.path.join(output_dir, f"orig_{img_file}"), enhanced_img)

        # ✅ Create augmented versions
        for i in range(AUGMENTATION_FACTOR):
            aug_img = augment_image(enhanced_img)
            np.save(os.path.join(output_dir, f"aug_{i}_{img_file}"), aug_img)

# 🚀 Dataset Balancing
def balance_dataset(base_dir):
    """Balance classes by oversampling minority classes"""
    for split in ['train', 'val']:
        counts = []
        class_dirs = []

        # ✅ Count files in each class
        for cls in CLASSES:
            cls_dir = os.path.join(base_dir, split, cls)
            if not os.path.exists(cls_dir):
                print(f"Class directory not found: {cls_dir}")
                continue

            n_files = len(os.listdir(cls_dir))
            counts.append(n_files)
            class_dirs.append(cls_dir)

        # ✅ Max class size for oversampling
        max_count = max(counts)

        # ✅ Resample and balance
        for i, (cls_dir, n) in enumerate(zip(class_dirs, counts)):
            if n == 0:
                continue

            files = os.listdir(cls_dir)
            if n < max_count:
                # Oversample to match max_count
                resampled = resample(files, 
                                    replace=True, 
                                    n_samples=max_count - n,
                                    random_state=42)

                for f in resampled:
                    src = os.path.join(cls_dir, f)
                    dst = os.path.join(cls_dir, f"bal_{f}")
                    shutil.copy(src, dst)

# 🚀 Main Execution
def main():
    """Main driver function"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ✅ Create enhanced and augmented dataset
    for split in ['train', 'val']:
        for cls in CLASSES:
            src_dir = os.path.join(BASE_DIR, split, cls)
            dest_dir = os.path.join(OUTPUT_DIR, split, cls)

            if not os.path.exists(src_dir):
                print(f"Skipping missing directory: {src_dir}")
                continue

            process_and_save(src_dir, dest_dir)

    # ✅ Balance classes
    balance_dataset(OUTPUT_DIR)

    print(f"\n🔥 New dataset created at: {OUTPUT_DIR}")
    print(f"🔥 Total augmentation: {AUGMENTATION_FACTOR}x")
    print("🔥 Data quality checks and balancing complete!")

# 🚀 Run
if __name__ == "__main__":
    main()
