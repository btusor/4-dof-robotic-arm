"""Red thresholding: BGR image -> binary mask, one white blob per red Smartie.
"""
import os
import cv2
from image_analysis import find_red_spots, red_threshold_mask, spots_mask, draw_spots


def threshold_red(img, debug_dir=None):
    """
    mask : uint8 image, 255 inside every detected red Smartie, 0 elsewhere
    spots: list of dicts from find_red_spots (center, area, contour, ...)
    """
    spots, candies = find_red_spots(img)
    mask = spots_mask(img.shape, spots)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "02_red_threshold_raw.png"), red_threshold_mask(img))
        cv2.imwrite(os.path.join(debug_dir, "03_all_candies.png"), draw_spots(img, candies))
        cv2.imwrite(os.path.join(debug_dir, "04_red_mask.png"), mask)
        cv2.imwrite(os.path.join(debug_dir, "05_red_overlay.png"), draw_spots(img, spots))
    return mask, spots
