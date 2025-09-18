# ====================================
# Install requirements (if needed)
# ====================================
!pip install opencv-python-headless matplotlib scikit-learn

# ====================================
# Imports
# ====================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from google.colab import files

# ====================================
# File Upload
# ====================================
uploaded = files.upload()
fname = list(uploaded.keys())[0]
print("Uploaded file:", fname)

# ====================================
# Dot detection
# ====================================
def detect_raw_dots(img, params=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)

    if params is None:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 500
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = False
        params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inv)

    dot_coords = np.array([(k.pt[0], k.pt[1]) for k in keypoints], dtype=np.float64)
    return dot_coords, gray


def merge_close_dots(dot_coords, eps=10):
    """Merge duplicates using DBSCAN clustering."""
    if len(dot_coords) == 0:
        return dot_coords
    clustering = DBSCAN(eps=eps, min_samples=1).fit(dot_coords)
    cluster_centers = []
    for label in sorted(set(clustering.labels_)):
        members = dot_coords[clustering.labels_ == label]
        cluster_centers.append(members.mean(axis=0))
    return np.array(cluster_centers, dtype=np.float64)


# ====================================
# Skeletonization
# ====================================
def get_skeleton(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)

    # Try OpenCV thinning if available
    try:
        skeleton = cv2.ximgproc.thinning(binary)
    except:
        # Fallback using morphological erosion
        size = np.size(binary)
        skel = np.zeros(binary.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        img = binary.copy()
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        skeleton = skel

    return skeleton


# ====================================
# Visualizations
# ====================================
def show_detected_dots(img, dot_coords, title="Detected Dots"):
    vis = img.copy()
    for (x, y) in dot_coords:
        cv2.circle(vis, (int(round(x)), int(round(y))), 5, (0, 200, 0), -1)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()


def show_skeleton(skeleton, dot_coords=None, title="Kolam Skeleton"):
    vis = np.ones((*skeleton.shape, 3), dtype=np.uint8) * 255
    vis[skeleton > 0] = (0, 0, 0)  # skeleton in black

    if dot_coords is not None:
        for (x, y) in dot_coords:
            cv2.circle(vis, (int(round(x)), int(round(y))), 5, (200, 0, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.imshow(vis)
    plt.axis("off")
    plt.title(title)
    plt.show()


# ====================================
# Run pipeline
# ====================================
img = cv2.imread(fname)
if img is None:
    raise FileNotFoundError(f"Could not open {fname}")

# 1. Dot detection
raw_dots, gray = detect_raw_dots(img)
dot_coords = merge_close_dots(raw_dots, eps=10)
print("Dots detected:", len(dot_coords))
show_detected_dots(img, dot_coords)

# 2. Skeleton
skeleton = get_skeleton(gray)
show_skeleton(skeleton, dot_coords, title="Kolam Skeleton")
