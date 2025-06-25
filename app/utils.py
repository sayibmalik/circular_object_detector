import cv2
import numpy as np
import uuid
import os
from typing import List, Dict
def segment_circles(image_path: str, output_dir: str) -> (List[Dict], str):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Hough Circle Detection
    # circles = cv2.HoughCircles(
    #     enhanced, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
    #     param1=100, param2=40, minRadius=20, maxRadius=80
    # )
    # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
    #                             param1=100, param2=50, minRadius=80, maxRadius=150)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                                param1=100, param2=50, minRadius=50, maxRadius=150)

    # Create mask for coin areas
    coin_mask = np.zeros((h, w), dtype=np.uint8)
    results = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x, y, r) in circles:
            region_mask = np.zeros_like(gray)
            cv2.circle(region_mask, (x, y), r, 255, -1)
            brightness = cv2.mean(gray, mask=region_mask)[0]
            if brightness < 100:
                continue

            uid = str(uuid.uuid4())
            bbox = [int(x - r), int(y - r), int(x + r), int(y + r)]
            results.append({
                "id": uid,
                "bbox": bbox,
                "centroid": [int(x), int(y)],
                "radius": int(r)
            })

            # Draw annotation
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(output, uid[:4], (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.circle(coin_mask, (x, y), r, 255, -1)

    # Alpha blend the blue background with 70% opacity
    blue_overlay = np.full_like(output, (255, 0, 0))  # BGR Blue
    alpha = 0.7

    # Create inverse mask for background
    inv_mask = cv2.bitwise_not(coin_mask)
    inv_mask_3ch = cv2.merge([inv_mask, inv_mask, inv_mask]) // 255

    # Convert to float for blending
    output_float = output.astype(np.float32)
    blue_float = blue_overlay.astype(np.float32)

    # Blend only background pixels
    blended = output_float * (1 - alpha) + blue_float * alpha
    final_image = output_float.copy()
    final_image[inv_mask_3ch == 1] = blended[inv_mask_3ch == 1]

    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    # Save annotated result
    annotated_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(annotated_path, final_image)

    return results, annotated_path