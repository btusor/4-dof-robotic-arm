import sys
import cv2
import numpy as np


def convert_to_hsv(img):
    # OpenCV images (cv2.imread / VideoCapture) are BGR, not RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def find_candies(img, s_min=45, min_area=1500, max_area=15000,
                 min_circularity=0.6, blur=5):
    """Segment all M&Ms from the paper background.

    Paper has saturation ~20, every M&M colour (even blue) is > 45, so a
    saturation threshold separates candies from paper. Each candy blob gets
    its median hue/saturation/value so it can be classified by colour.

    Returns a list of dicts: center (x, y), radius, area, circularity,
    hue (relative to red, -90..89), sat, val, bbox, contour.
    """
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    hsv = convert_to_hsv(img)

    mask = cv2.inRange(hsv[:, :, 1], s_min, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candies = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / perimeter ** 2 if perimeter else 0
        if circularity < min_circularity:
            continue

        # Median colour of the inner part of the blob (avoid edge pixels)
        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        blob = cv2.erode(blob, kernel)
        px = hsv[blob > 0]
        h = px[:, 0].astype(int)
        h = np.where(h >= 90, h - 180, h)  # unwrap: red is around 0

        m = cv2.moments(c)
        _, radius = cv2.minEnclosingCircle(c)
        candies.append({
            "center": (m["m10"] / m["m00"], m["m01"] / m["m00"]),
            "radius": radius, "area": area, "circularity": circularity,
            "hue": float(np.median(h)), "sat": float(np.median(px[:, 1])),
            "val": float(np.median(px[:, 2])),
            "bbox": cv2.boundingRect(c), "contour": c,
        })
    return candies


def find_red_spots(img, hue_lo=-5, hue_hi=4, s_min=110, fill_min=0.4, **kwargs):
    """Find red M&Ms in a BGR image.

    Hue is relative to red (OpenCV hue wraps at 180), so hue_lo=-5,
    hue_hi=4 means OpenCV hues 175..179 and 0..4.
    Median values measured on snapshot_1_0.png:
        red      hue -3 .. -1   sat 130-145
        orange   hue +8 .. +9   sat 135-155   (closest to red)
        brown    hue +7 .. +8   sat  90-110
        pink     hue -9         sat ~100
    Retune if the lighting changes.

    fill_min: minimum fraction of each candy area that must be red in the
    pixel threshold (red_threshold_mask). Rejects candies that are only
    partly red, e.g. the red-ish rim on orange/brown ones. Measured:
    red ~0.66 (highlights/shadow edge keep it below 1), others <= 0.06.
    Each candy dict gets a "red_fill" value.

    Returns (spots, candies): spots = the red candies, candies = all
    candies found (useful for debugging thresholds).
    """
    candies = find_candies(img, **kwargs)
    threshold = red_threshold_mask(img, hue_lo, hue_hi, s_min)
    for c in candies:
        area = np.zeros(threshold.shape, np.uint8)
        cv2.drawContours(area, [c["contour"]], -1, 255, -1)
        c["red_fill"] = float((threshold[area > 0] > 0).mean())
    spots = [c for c in candies
             if hue_lo <= c["hue"] <= hue_hi and c["sat"] >= s_min
             and c["red_fill"] >= fill_min]
    spots.sort(key=lambda s: s["area"], reverse=True)
    return spots, candies


def draw_spots(img, spots):
    """Return a copy of img with the spots outlined and numbered."""
    out = img.copy()
    for i, s in enumerate(spots):
        cv2.drawContours(out, [s["contour"]], -1, (0, 255, 0), 2)
        cx, cy = map(int, s["center"])
        cv2.circle(out, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(out, str(i + 1), (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return out


def red_threshold_mask(img, hue_lo=-5, hue_hi=4, s_min=110, v_min=80, blur=5):
    """Pixel-wise binary threshold of red: 255 = red pixel, 0 = not red.

    Uses the same hue/saturation window as find_red_spots. This is the raw
    threshold, so highlights and edges of red M&Ms may show as holes, and a
    few orange/pink pixels may appear as speckles.
    """
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    hsv = convert_to_hsv(img)
    mask = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([hue_hi, 255, 255]))
    if hue_lo < 0:
        mask |= cv2.inRange(hsv, np.array([180 + hue_lo, s_min, v_min]), np.array([179, 255, 255]))
    return mask


def spots_mask(shape, spots):
    """Binary image with the detected spots filled white (clean result)."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.drawContours(mask, [s["contour"] for s in spots], -1, 255, -1)
    return mask


def save_red_masks(img, out_prefix, spots=None):
    """Save the red binary images as PNGs and return their paths.

    <out_prefix>_red_threshold.png          : raw pixel threshold
    <out_prefix>_red_threshold_filtered.png : threshold kept only inside the
                                              candies that passed (fill_min etc.)
    <out_prefix>_red_mask.png               : detected red M&Ms filled white
    """
    if spots is None:
        spots, _ = find_red_spots(img)
    paths = {
        "threshold": f"{out_prefix}_red_threshold.png",
        "filtered": f"{out_prefix}_red_threshold_filtered.png",
        "mask": f"{out_prefix}_red_mask.png",
    }
    threshold = red_threshold_mask(img)
    mask = spots_mask(img.shape, spots)
    cv2.imwrite(paths["threshold"], threshold)
    cv2.imwrite(paths["filtered"], threshold & mask)
    cv2.imwrite(paths["mask"], mask)
    return paths


if __name__ == "__main__":
    # Usage: python image_analysis.py path/to/image.png
    path = sys.argv[1]
    img = cv2.imread(path)
    spots, _ = find_red_spots(img)
    for i, s in enumerate(spots):
        print(f"Spot {i+1}: center=({s['center'][0]:.0f}, {s['center'][1]:.0f}) area={s['area']:.0f} px")
    prefix = path.rsplit(".", 1)[0]
    cv2.imwrite(prefix + "_red.png", draw_spots(img, spots))
    print("Saved", prefix + "_red.png")
    for p in save_red_masks(img, prefix, spots).values():
        print("Saved", p)
