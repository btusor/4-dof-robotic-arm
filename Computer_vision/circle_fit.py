"""Circle fitting: binary mask -> smallest enclosing circle of every white blob."""
import csv
import os
import cv2

def fit_circles(mask, img=None, debug_dir=None):
    """Return a list of circles (u, v, r) in pixels, largest first.

    (u, v) is the circle centre, r the radius of the smallest circle that
    contains the whole blob (cv2.minEnclosingCircle).

    If debug_dir is given, saves:
        06_circles.png : fitted circles drawn on img (or on the mask)
        circles.csv    : index, u, v, r of every circle
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    circles = []
    
    for c in contours:
        (u, v), r = cv2.minEnclosingCircle(c)
        circles.append((u, v, r))
    circles.sort(key=lambda c: c[2], reverse=True)

    if debug_dir:
        out = img.copy() if img is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for i, (u, v, r) in enumerate(circles):
            center = (int(round(u)), int(round(v)))
            cv2.circle(out, center, int(round(r)), (0, 255, 0), 2)
            cv2.circle(out, center, 3, (255, 0, 0), -1)
            cv2.putText(out, f"{i+1}: r={r:.1f}px", (center[0] + int(r) + 5, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "06_circles.png"), out)

        with open(os.path.join(debug_dir, "circles.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "u_px", "v_px", "r_px"])
            for i, (u, v, r) in enumerate(circles):
                writer.writerow([i + 1, f"{u:.2f}", f"{v:.2f}", f"{r:.2f}"])
    return circles
