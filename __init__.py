import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def quantify_pigmentation(image_path, show_plot=True):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return 0

    # Resize for consistent processing speed
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height))

    # --- EDIT: Crop top 30% (increased to be safer against knot interference) ---
    crop_h = int(height * 0.30)
    img = img[crop_h:height, 0:width]

    # 2. Convert to HSV Color Space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Define Color Range
    # --- MAJOR EDIT: AGGRESSIVE FILTERING ---
    # We define strict "Red/Pink" and completely IGNORE "Orange/Brown"
    # Saturation (s_min) raised to 90 (Must be very colorful)
    # Value (v_min) raised to 90 (Must be bright, ignores dark wet dirt)

    s_min = 90
    v_min = 90

    # Range 1: 0 to 8 (True Red only. Stops before Orange/Brown starts)
    lower_red1 = np.array([0, s_min, v_min])
    upper_red1 = np.array([8, 255, 255])

    # Range 2: 170 to 180 (Pink/Red)
    lower_red2 = np.array([170, s_min, v_min])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    pigment_mask = cv2.bitwise_or(mask1, mask2)

    # 4. Define Color Range for Substrate (Bag Area)
    # Standard mask to find the bag area
    lower_bag = np.array([0, 20, 20])
    upper_bag = np.array([180, 255, 255])
    bag_mask_base = cv2.inRange(hsv, lower_bag, upper_bag)

    # Combine to get total biological area
    total_bag_mask = cv2.bitwise_or(bag_mask_base, pigment_mask)

    # 5. Noise Cleanup
    kernel = np.ones((5, 5), np.uint8)
    pigment_mask = cv2.morphologyEx(pigment_mask, cv2.MORPH_OPEN, kernel)
    total_bag_mask = cv2.morphologyEx(total_bag_mask, cv2.MORPH_OPEN, kernel)

    # --- EDIT: Increased Blob Size Filter ---
    # Raised to 150 to delete larger patches of glare/noise
    contours, _ = cv2.findContours(pigment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 150:  # If blob is too small
            cv2.drawContours(pigment_mask, [cnt], -1, 0, -1)  # Draw black (remove it)

    # 6. Calculate Area
    pigment_pixels = cv2.countNonZero(pigment_mask)
    total_bag_pixels = cv2.countNonZero(total_bag_mask)

    if total_bag_pixels == 0:
        ratio = 0
    else:
        ratio = (pigment_pixels / total_bag_pixels) * 100

    # 7. Visualization
    if show_plot:
        vis_img = img.copy()
        # Changed contour color to Red for better visibility against brown
        contours, _ = cv2.findContours(pigment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original (Cropped)")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pigment_mask, cmap='gray')
        plt.title("Pigment Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Overlay (Cov: {ratio:.2f}%)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return ratio


# --- RUN THE ANALYSIS ---
image_files = []
results = []
days = []

print("Starting Analysis...")
# Assuming you have images named day1.jpeg to day15.jpeg
for i in range(1, 16):
    file_name = f"fungal_growth_images/day{i}.jpeg"

    # Only process if file actually exists
    if os.path.exists(file_name):
        days.append(i)
        image_files.append(file_name)
        print(f"Processing {file_name}...")
        percentage = quantify_pigmentation(file_name)
        results.append(percentage)
        print(f"Coverage: {percentage:.2f}%")

# Plotting the Growth Curve
if len(results) > 0:
    plt.figure(figsize=(8, 5))
    plt.plot(days, results, marker='o', linestyle='-', color='r')
    plt.title("Fungal Pigmentation Growth Over Time")
    plt.xlabel("Day")
    plt.ylabel("Pigmentation Coverage (%)")
    plt.grid(True)
    plt.show()
else:
    print("No images found to plot.")