!pip install --quiet opencv-python-headless matplotlib numpy scipy scikit-learn shapely svgwrite

from google.colab import files
import cv2, numpy as np, matplotlib.pyplot as plt, math, svgwrite
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from scipy import interpolate
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import unary_union, snap
from IPython.display import SVG, display

uploaded = files.upload()
image_path = list(uploaded.keys())[0]
img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_bgr is None:
    print("Error: Could not load image from", image_path)
else:
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Original Uploaded Image")
    plt.show()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def detect_features(gray_img, detection_params):
        clahe = cv2.createCLAHE(clipLimit=detection_params['clahe_clip'], tileGridSize=detection_params['clahe_grid'])
        gray_eq = clahe.apply(gray_img)
        blur = cv2.GaussianBlur(gray_eq, detection_params['blur_kernel'], 0)
        edges = cv2.Canny(blur, detection_params['canny_low'], detection_params['canny_high'], apertureSize=3)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        quads = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, detection_params['poly_epsilon'] * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > detection_params['min_quad_area']:
                    quads.append(approx.reshape(4,2))

        circles = cv2.HoughCircles(gray_eq, cv2.HOUGH_GRADIENT, dp=detection_params['circle_dp'],
                                   minDist=detection_params['circle_mindist'],
                                   param1=detection_params['circle_param1'],
                                   param2=detection_params['circle_param2'],
                                   minRadius=detection_params['circle_minr'],
                                   maxRadius=detection_params['circle_maxr'])
        dot_points = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                dot_points.append((x, y))

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=detection_params['line_threshold'],
                                minLineLength=detection_params['line_minlen'],
                                maxLineGap=detection_params['line_maxgap'])
        line_segments = []
        if lines is not None:
            pts = []
            for l in lines:
                x1,y1,x2,y2 = l[0]
                pts.append([x1,y1]); pts.append([x2,y2])
            pts = np.array(pts)
            remaining = pts.copy()
            while len(remaining) > 20:
                X = remaining[:,0].reshape(-1,1); y = remaining[:,1]
                ransac = RANSACRegressor(min_samples=0.25, residual_threshold=3.0)
                try: ransac.fit(X, y)
                except: break
                in_mask = ransac.inlier_mask_; in_pts = remaining[in_mask]
                if len(in_pts) < 10: break
                x_min, y_min = in_pts.min(axis=0); x_max, y_max = in_pts.max(axis=0)
                seg = ((int(x_min), int(y_min)), (int(x_max), int(y_max)))
                line_segments.append(seg)
                remaining = remaining[~in_mask]
                if len(line_segments) > 20: break

        curve_splines = []
        for idx, cnt in enumerate(contours):
            alen = cv2.arcLength(cnt, closed=True)
            if alen < detection_params['min_curve_len']: continue
            pts = cnt.reshape(-1,2).astype(float)
            eps = 1.0; approx = cv2.approxPolyDP(pts.astype(np.float32), epsilon=eps, closed=True)
            pts_s = approx.reshape(-1,2).astype(float)
            if pts_s.shape[0] < 4: pts_s = pts
            try:
                tck, u = interpolate.splprep([pts_s[:,0], pts_s[:,1]], s=1.0, k=3, per=True)
                unew = np.linspace(0,1,200); out = interpolate.splev(unew, tck)
                spline_pts = np.vstack(out).T
                curve_splines.append({'pts': spline_pts})
            except:
                curve_splines.append({'pts': pts_s})

        return quads, dot_points, line_segments, curve_splines

    detection_configs = [
        {'clahe_clip': 3.0, 'clahe_grid': (8,8), 'blur_kernel': (5,5), 'canny_low': 50, 'canny_high': 150,
         'poly_epsilon': 0.02, 'min_quad_area': 80, 'circle_dp': 1.2, 'circle_mindist': 8,
         'circle_param1': 60, 'circle_param2': 20, 'circle_minr': 2, 'circle_maxr': 20,
         'line_threshold': 60, 'line_minlen': 25, 'line_maxgap': 12, 'min_curve_len': 20},
        {'clahe_clip': 2.0, 'clahe_grid': (6,6), 'blur_kernel': (3,3), 'canny_low': 30, 'canny_high': 120,
         'poly_epsilon': 0.015, 'min_quad_area': 60, 'circle_dp': 1.5, 'circle_mindist': 6,
         'circle_param1': 50, 'circle_param2': 15, 'circle_minr': 1, 'circle_maxr': 25,
         'line_threshold': 45, 'line_minlen': 20, 'line_maxgap': 15, 'min_curve_len': 15},
        {'clahe_clip': 4.0, 'clahe_grid': (10,10), 'blur_kernel': (7,7), 'canny_low': 70, 'canny_high': 180,
         'poly_epsilon': 0.025, 'min_quad_area': 100, 'circle_dp': 1.0, 'circle_mindist': 10,
         'circle_param1': 70, 'circle_param2': 25, 'circle_minr': 3, 'circle_maxr': 15,
         'line_threshold': 80, 'line_minlen': 30, 'line_maxgap': 8, 'min_curve_len': 25}
    ]

    all_quads, all_dots, all_lines, all_curves = [], [], [], []

    for config in detection_configs:
        quads, dot_points, line_segments, curve_splines = detect_features(gray, config)
        all_quads.extend(quads)
        all_dots.extend(dot_points)
        all_lines.extend(line_segments)
        all_curves.extend(curve_splines)

    quads = all_quads
    dot_points = np.array(all_dots, dtype=float) if all_dots else np.array([])
    line_segments = all_lines
    curve_splines = all_curves

    lattice_points = []
    if len(dot_points) > 0:
        clustering = DBSCAN(eps=6, min_samples=1).fit(dot_points)
        labels = clustering.labels_
        for lab in np.unique(labels):
            pts = dot_points[labels == lab]
            centroid = pts.mean(axis=0)
            lattice_points.append(tuple(centroid))
    lattice_points = np.array(lattice_points)

    def snap_to_grid(pts, grid_tol=6):
        if len(pts) < 2: return pts
        dists = np.sqrt(((pts[:,None,:]-pts[None,:,:])**2).sum(-1))
        dists[dists==0] = np.nan
        median = np.nanmedian(dists)
        snapped = np.round(pts/median)*median
        uniq = np.unique(snapped.round(2), axis=0)
        return uniq
    lattice_snapped = snap_to_grid(lattice_points) if len(lattice_points)>0 else lattice_points

    h, w = img_bgr.shape[:2]
    dwg = svgwrite.Drawing("kolam_clean.svg", size=(f"{w}px", f"{h}px"))

    filtered_dots = []
    for (x, y) in lattice_snapped:
        is_inside_quad = False
        for q in quads:
            if cv2.pointPolygonTest(q.astype(np.float32), (x, y), False) >= 0:
                is_inside_quad = True
                break
        if not is_inside_quad:
            filtered_dots.append((x, y))

    for (x,y) in filtered_dots:
        dwg.add(dwg.circle(center=(float(x), float(y)), r=1.5,
                           stroke='gray', stroke_width=0.5, fill='none'))

    for seg in line_segments:
        (x1,y1),(x2,y2) = seg
        dwg.add(dwg.line(start=(x1,y1), end=(x2,y2),
                         stroke='black', stroke_width=2))

    for c in curve_splines:
        pts = c['pts']
        if pts.shape[0] < 2:
            continue
        path_d = "M %f %f " % (pts[0,0], pts[0,1])
        for p in pts[1:]:
            path_d += "L %f %f " % (p[0], p[1])
        dwg.add(dwg.path(d=path_d, stroke='black', fill='none', stroke_width=2,
                         stroke_linecap='round', stroke_linejoin='round'))

    for q in quads:
        pts = [(float(x), float(y)) for x,y in q]
        dwg.add(dwg.polygon(points=pts, stroke='black', fill='none', stroke_width=1.75))

    svg_string = dwg.tostring()
    display(SVG(svg_string))
    print("Kolam pattern detection completed and displayed above.")