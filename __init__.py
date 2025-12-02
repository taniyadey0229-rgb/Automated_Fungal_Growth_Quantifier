import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def quantify_pigmentation(image_path, day_label, output_folder):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        return 0, None

    # Resize
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height))

    # Crop top 30%
    crop_h = int(height * 0.30)
    img = img[crop_h:height, 0:width]

    # 2. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Define Color Range ("Pink Backdoor" Strategy)
    s_min = 80
    v_min = 80

    lower_red1 = np.array([0, s_min, v_min])
    upper_red1 = np.array([9, 255, 255])
    lower_red2 = np.array([155, s_min, v_min])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    pigment_mask = cv2.bitwise_or(mask1, mask2)

    # 4. Substrate Mask
    lower_bag = np.array([0, 20, 20])
    upper_bag = np.array([180, 255, 255])
    bag_mask_base = cv2.inRange(hsv, lower_bag, upper_bag)
    total_bag_mask = cv2.bitwise_or(bag_mask_base, pigment_mask)

    # 5. Noise Cleanup
    kernel = np.ones((5, 5), np.uint8)
    pigment_mask = cv2.morphologyEx(pigment_mask, cv2.MORPH_OPEN, kernel)
    total_bag_mask = cv2.morphologyEx(total_bag_mask, cv2.MORPH_OPEN, kernel)

    # Blob Size Filter
    contours, _ = cv2.findContours(pigment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 120:
            cv2.drawContours(pigment_mask, [cnt], -1, 0, -1)

    # 6. Calculate Area
    pigment_pixels = cv2.countNonZero(pigment_mask)
    total_bag_pixels = cv2.countNonZero(total_bag_mask)

    if total_bag_pixels == 0:
        ratio = 0
    else:
        ratio = (pigment_pixels / total_bag_pixels) * 100

    # 7. Visualization PREPARATION (Do not show, just save)
    vis_img = img.copy()
    contours, _ = cv2.findContours(pigment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)

    # Add text label to the image itself for the report
    cv2.putText(vis_img, f"{day_label}: {ratio:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- SAVE INDIVIDUAL DETAILED REPORT TO DISK ---
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Analysis for {day_label}", fontsize=14)
    plt.subplot(1, 3, 1);
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    plt.title("Original")
    plt.subplot(1, 3, 2);
    plt.imshow(pigment_mask, cmap='gray');
    plt.title("Mask")
    plt.subplot(1, 3, 3);
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB));
    plt.title(f"Overlay")

    # Save file instead of showing it
    save_path = os.path.join(output_folder, f"{day_label.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()  # Close figure to free memory

    return ratio, vis_img


# --- RUN THE ANALYSIS ---
output_folder = "analysis_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = []
results = []
days = []
report_images = []  # Store images for the final grid

print(f"Starting Analysis... (Saving details to '{output_folder}/')")

for i in range(1, 16):
    file_name = f"fungal_growth_images/day{i}.jpeg"

    if os.path.exists(file_name):
        print(f"Processing Day {i}...", end='\r')  # \r prints on same line

        # We pass the output folder to save individual images there
        percentage, overlay_img = quantify_pigmentation(file_name, f"Day {i}", output_folder)

        days.append(i)
        results.append(percentage)
        report_images.append(overlay_img)
    else:
        # If a day is missing, add a blank placeholder so the grid doesn't break
        pass

print(f"\nAnalysis Complete. Generating Summary Report...")

# --- 1. PLOT GROWTH CURVE ---
plt.figure(figsize=(10, 6))
plt.plot(days, results, marker='o', linestyle='-', color='r', linewidth=2)
plt.title("Fungal Pigmentation Growth Over Time")
plt.xlabel("Day")
plt.ylabel("Pigmentation Coverage (%)")
plt.grid(True)
plt.show()  # This is the first popup

# --- 2. GENERATE SINGLE LARGE VISUAL REPORT (GRID) ---
# Calculate grid size (e.g., if 15 images, 3 rows of 5)
num_imgs = len(report_images)
cols = 5
rows = math.ceil(num_imgs / cols)

plt.figure(figsize=(15, 3 * rows))  # Dynamic height based on rows
plt.suptitle("Visual Progression Report", fontsize=16)

for idx, img in enumerate(report_images):
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.tight_layout()
plt.show()  # This is the second popup (The "Contact Sheet")