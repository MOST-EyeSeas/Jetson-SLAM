#!/usr/bin/env python3
import os
import glob
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def compute_focus_measure(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def edge_density(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    return float(edges.mean()) / 255.0, edges


def detect_keypoints(img_gray, max_features=2000):
    fast = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
    kps = fast.detect(img_gray, None)
    # Cap for speed in descriptors
    kps = sorted(kps, key=lambda k: -k.response)[:max_features]
    orb = cv2.ORB_create(nfeatures=min(max_features, 2000), scaleFactor=1.2, nlevels=8)
    kps, des = orb.compute(img_gray, kps)
    return kps, des


def match_parallax(img1_gray, img2_gray):
    k1, d1 = detect_keypoints(img1_gray)
    k2, d2 = detect_keypoints(img2_gray)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return 0, 0.0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if not matches:
        return 0, 0.0, 0.0
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    # Estimate essential/homography metrics
    F, maskF = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    H, maskH = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.0)
    inliersF = int(maskF.sum()) if maskF is not None else 0
    inliersH = int(maskH.sum()) if maskH is not None else 0
    # Compute average pixel displacement
    disp = np.linalg.norm(pts1 - pts2, axis=1)
    mean_disp = float(np.mean(disp)) if disp.size > 0 else 0.0
    return len(matches), mean_disp, (inliersF, inliersH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='Recording root (contains images/ and times.txt)')
    ap.add_argument('--stride', type=int, default=5, help='Frame step for parallax sampling')
    ap.add_argument('--limit', type=int, default=300, help='Max frames to analyze')
    args = ap.parse_args()

    img_dir = os.path.join(args.run_dir, 'images')
    frames = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if not frames:
        print('No frames found in', img_dir)
        return
    frames = frames[:args.limit]

    focus_vals, key_counts, edge_vals = [], [], []

    print('Analyzing single-frame metrics...')
    for f in tqdm(frames):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        focus_vals.append(compute_focus_measure(img))
        kps, _ = detect_keypoints(img)
        key_counts.append(len(kps))
        ed, _ = edge_density(img)
        edge_vals.append(ed)

    print('\nAnalyzing parallax/matches...')
    parallax = []
    inlier_stats = []
    for i in tqdm(range(0, len(frames) - args.stride, args.stride)):
        img1 = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frames[i + args.stride], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        nm, mean_disp, (inF, inH) = match_parallax(img1, img2)
        parallax.append(mean_disp)
        inlier_stats.append((inF, inH))

    def summary(name, arr):
        arr = np.array(arr, dtype=float)
        return f"{name}: mean={arr.mean():.2f} median={np.median(arr):.2f} min={arr.min():.2f} max={arr.max():.2f}"

    print('\n--- Summary ---')
    if focus_vals:
        print(summary('Laplacian focus (blur)', focus_vals))
    if key_counts:
        print(summary('ORB keypoints per frame', key_counts))
    if edge_vals:
        print(summary('Edge density (0-1)', edge_vals))
    if parallax:
        print(summary('Inter-frame pixel displacement', parallax))
        inF = [x[0] for x in inlier_stats]
        inH = [x[1] for x in inlier_stats]
        print(summary('Fundamental inliers', inF))
        print(summary('Homography inliers', inH))

    # Guidance messages
    if np.median(key_counts) < 300:
        print('Few keypoints detected; consider lower FAST thresholds, add texture, or CLAHE.')
    if np.median(parallax) < 1.5:
        print('Low parallax between frames; avoid pure yaw, add translation during initialization.')

if __name__ == '__main__':
    main()
