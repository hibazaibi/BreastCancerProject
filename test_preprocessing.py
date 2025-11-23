"""
Test preprocessing pipeline and visualize results
Place your test mammogram as 'test_mammogram.jpg' or 'test_mammogram.png'
"""
"""
Test preprocessing pipeline and visualize results
Place your test mammogram as 'test_mammogram.jpg' or 'test_mammogram.png'
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Import preprocessing function
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess
except ImportError:
    try:
        from keras.applications.efficientnet import preprocess_input as efn_preprocess
    except ImportError:
        print("ERROR: Cannot import EfficientNet preprocessing!")
        print("Install tensorflow: pip install tensorflow")
        exit(1)

print(f"✓ TensorFlow version: {tf.__version__}")


def simple_preprocess(img):
    """Auto-crop to remove background - ROBUST VERSION"""
    print(f"\n   🔍 Analyzing image for cropping...")
    print(f"      Image range: [{img.min()}, {img.max()}]")
    print(f"      Image mean: {img.mean():.1f}")

    # Calculate a dynamic threshold based on image statistics
    # Use Otsu's method or percentile-based threshold

    # Method 1: Try Otsu's threshold
    _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 2: Try percentile-based (keep pixels above 1st percentile)
    threshold_percentile = np.percentile(img[img > 0], 1) if np.any(img > 0) else 10
    _, binary_percentile = cv2.threshold(img, threshold_percentile, 255, cv2.THRESH_BINARY)

    # Method 3: Simple threshold
    _, binary_simple = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    # Try each method and pick the best crop
    methods = [
        ("Otsu", binary_otsu),
        (f"Percentile (thresh={threshold_percentile:.1f})", binary_percentile),
        ("Simple (thresh=10)", binary_simple)
    ]

    best_crop = img
    best_reduction = 0
    best_method = "None"

    for method_name, binary in methods:
        coords = cv2.findNonZero(binary)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Calculate reduction
            original_size = img.shape[0] * img.shape[1]
            cropped_size = w * h
            reduction = (1 - cropped_size / original_size) * 100

            print(f"      {method_name}: {reduction:.1f}% reduction, bbox=({w}x{h})")

            # We want significant reduction (10-80%)
            if 10 < reduction < 80 and reduction > best_reduction:
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2 * padding)
                h = min(img.shape[0] - y, h + 2 * padding)
                best_crop = img[y:y + h, x:x + w]
                best_reduction = reduction
                best_method = method_name

    if best_reduction > 0:
        print(f"   ✓ Best method: {best_method} ({best_reduction:.1f}% reduction)")
    else:
        print(f"   ⚠️  No effective crop found - using original image")
        print(f"      This might affect prediction accuracy!")

    return best_crop


# Find test image
test_files = ['test_mammogram.jpg', 'test_mammogram.png', 'test.jpg', 'mammogram.jpg']
test_file = None

for fname in test_files:
    if os.path.exists(fname):
        test_file = fname
        break

# Also check uploads folder
if test_file is None and os.path.exists('uploads'):
    upload_files = os.listdir('uploads')
    if upload_files:
        test_file = os.path.join('uploads', upload_files[0])

if test_file is None:
    print("❌ No test image found!")
    print("Please place a mammogram image as 'test_mammogram.jpg'")
    exit(1)

print(f"✓ Found test image: {test_file}")
print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE TEST")
print("=" * 60)

# Step 1: Load image
img_original = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
print(f"\n1. Original Image:")
print(f"   Shape: {img_original.shape}")
print(f"   Range: [{img_original.min()}, {img_original.max()}]")
print(f"   Mean: {img_original.mean():.1f}")

# Step 2: Auto-crop
img_cropped = simple_preprocess(img_original.copy())
print(f"\n2. After Auto-Crop:")
print(f"   Shape: {img_cropped.shape}")
print(f"   Cropped: {img_original.shape} → {img_cropped.shape}")
crop_percent = (img_cropped.size / img_original.size) * 100
print(f"   Kept: {crop_percent:.1f}% of original")

# Step 3: Resize
img_resized = cv2.resize(img_cropped, (512, 512))
print(f"\n3. After Resize to 512x512:")
print(f"   Shape: {img_resized.shape}")
print(f"   Range: [{img_resized.min()}, {img_resized.max()}]")

# Step 4: Convert to RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
print(f"\n4. After RGB Conversion:")
print(f"   Shape: {img_rgb.shape}")
print(f"   All channels same? {np.array_equal(img_rgb[:, :, 0], img_rgb[:, :, 1])}")

# Step 5: EfficientNet preprocessing
img_preprocessed = efn_preprocess(img_rgb.astype(np.float32))
print(f"\n5. After EfficientNet Preprocessing:")
print(f"   Shape: {img_preprocessed.shape}")
print(f"   Range: [{img_preprocessed.min():.3f}, {img_preprocessed.max():.3f}]")
print(f"   Mean: {img_preprocessed.mean():.3f}")
print(f"   Std: {img_preprocessed.std():.3f}")

# Expected range for EfficientNet: roughly -1 to +1
if img_preprocessed.min() < -2 or img_preprocessed.max() > 2:
    print("   ⚠️  WARNING: Values outside expected range!")
else:
    print("   ✓ Values look correct for EfficientNet")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16, fontweight='bold')

# Original
axes[0, 0].imshow(img_original, cmap='gray')
axes[0, 0].set_title(f'1. Original\n{img_original.shape}')
axes[0, 0].axis('off')

# Cropped
axes[0, 1].imshow(img_cropped, cmap='gray')
axes[0, 1].set_title(f'2. Auto-Cropped\n{img_cropped.shape}')
axes[0, 1].axis('off')

# Resized
axes[0, 2].imshow(img_resized, cmap='gray')
axes[0, 2].set_title(f'3. Resized\n{img_resized.shape}')
axes[0, 2].axis('off')

# RGB (showing one channel since all are same)
axes[1, 0].imshow(img_rgb[:, :, 0], cmap='gray')
axes[1, 0].set_title(f'4. RGB (one channel)\n{img_rgb.shape}')
axes[1, 0].axis('off')

# Preprocessed (normalize to 0-255 for visualization)
img_viz = img_preprocessed[:, :, 0]
img_viz = ((img_viz - img_viz.min()) / (img_viz.max() - img_viz.min()) * 255).astype(np.uint8)
axes[1, 1].imshow(img_viz, cmap='gray')
axes[1, 1].set_title(f'5. After Preprocessing\n(normalized for display)')
axes[1, 1].axis('off')

# Histogram
axes[1, 2].hist(img_preprocessed.ravel(), bins=50, color='blue', alpha=0.7)
axes[1, 2].set_title('Preprocessed Values Distribution')
axes[1, 2].set_xlabel('Pixel Value')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].axvline(x=0, color='red', linestyle='--', label='Zero')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved as 'preprocessing_visualization.png'")
plt.show()

print("\n" + "=" * 60)
print("✅ PREPROCESSING TEST COMPLETE")
print("=" * 60)
print("\nCheck the visualization to see if cropping worked correctly.")
print("The cropped image should show just the breast tissue without black background.")
