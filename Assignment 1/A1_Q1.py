# A1_Q1.py — OpenCV logo (single image, flat/triangular cut edges, keep your directions)
import cv2, numpy as np, math

# ---------- Canvas ----------
W, H = 400, 420
img = np.full((H, W, 3), 255, np.uint8)

# ---------- Colors (BGR) ----------
RED   = (0, 0, 255)
GREEN = (0, 200, 0)
BLUE  = (255, 0, 0)
BLACK = (0, 0, 0)

# ---------- Geometry ----------
cx, cy = W // 2, 160
R      = 60         # outer radius
TH     = 35         # ring thickness
GAP_DEG = 60        # opening width
off    = 78         # distance of each ring center from cluster center

def deg2rad(a): return a * math.pi / 180.0

def draw_ring_flat(dst, center, color, start_angle):
    """
    Draw ring with flat/triangular gap edges (no rounded nubs).
    """
    # 1) annulus mask
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, center, R, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(mask, center, R - TH, 0, -1, lineType=cv2.LINE_AA)

    # 2) carve wedge for the gap (two straight rays from the center)
    a0 = start_angle % 360
    a1 = a0 + GAP_DEG
    L = R + 3*TH
    p0 = center
    p1 = (int(center[0] + L * math.cos(deg2rad(a0))),
          int(center[1] + L * math.sin(deg2rad(a0))))
    p2 = (int(center[0] + L * math.cos(deg2rad(a1))),
          int(center[1] + L * math.sin(deg2rad(a1))))
    cv2.fillConvexPoly(mask, np.array([p0, p1, p2], np.int32), 0, lineType=cv2.LINE_AA)

    # 3) paint color through mask
    colored = np.zeros_like(dst); colored[:] = color
    dst[:] = np.where(mask[..., None] == 255, colored, dst)

# --- Centers (triangular layout) ---
red_center   = (cx, cy - off)  # top
green_center = (cx - int(off * math.cos(math.radians(30))),
                cy + int(off * math.sin(math.radians(30))))  # bottom-left
blue_center  = (cx + int(off * math.cos(math.radians(30))),
                cy + int(off * math.sin(math.radians(30))))  # bottom-right

# --- Keep EXACT directions you liked ---
RED_START   = 420  # red opens downward
GREEN_START = 300  # green opens up-right
BLUE_START  = 240  # blue opens up-left

# --- Draw (flat edges) ---
draw_ring_flat(img, red_center,   RED,   RED_START)
draw_ring_flat(img, green_center, GREEN, GREEN_START)
draw_ring_flat(img, blue_center,  BLUE,  BLUE_START)

# --- Wordmark (closer to shapes) ---
bottom_of_rings = max(green_center[1], blue_center[1]) + R  # lowest point of rings
margin = 60  # distance between rings and text (make smaller for closer)
cv2.putText(
    img, "OpenCV",
    (cx - 100, int(bottom_of_rings + margin)),   # moved up relative to shapes
    cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 3, cv2.LINE_AA
)


# --- Save & show a single image ---
cv2.imwrite("opencv_logo_final.png", img)
cv2.imshow("OpenCV Logo", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("✅ Saved: opencv_logo_final.png")
