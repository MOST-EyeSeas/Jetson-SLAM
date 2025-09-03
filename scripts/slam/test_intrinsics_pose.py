#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
import cv2

K_CANDIDATES = {
    'K_181x153': np.array([[181.12027, 0.0, 319.31663],
                           [0.0, 152.88830, 179.86583],
                           [0.0, 0.0, 1.0]], dtype=np.float64),
    'K_525x525': np.array([[525.0, 0.0, 320.0],
                           [0.0, 525.0, 180.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
}

def detect_and_match(img1, img2):
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d1, d2)
    if not m:
        return None
    m = sorted(m, key=lambda x: x.distance)[:800]
    pts1 = np.float32([k1[x.queryIdx].pt for x in m])
    pts2 = np.float32([k2[x.trainIdx].pt for x in m])
    return pts1, pts2


def evaluate_pair(img1, img2, K):
    dm = detect_and_match(img1, img2)
    if dm is None:
        return None
    pts1, pts2 = dm
    E, inl = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or inl is None:
        return None
    inliers = int(inl.sum())
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    mask = mask.ravel().astype(bool)
    inliers_pose = int(mask.sum())
    # Epipolar reprojection error statistics
    pts1n = cv2.undistortPoints(pts1.reshape(-1,1,2), K, None).reshape(-1,2)
    pts2n = cv2.undistortPoints(pts2.reshape(-1,1,2), K, None).reshape(-1,2)
    # l2 = E x1
    x1h = np.hstack([pts1n, np.ones((pts1n.shape[0],1))])
    x2h = np.hstack([pts2n, np.ones((pts2n.shape[0],1))])
    l2 = (E @ x1h.T).T
    num = np.abs(np.sum(l2 * x2h, axis=1))
    den = np.sqrt(l2[:,0]**2 + l2[:,1]**2) + 1e-8
    epi_err = num/den
    return {
        'inliers_E': inliers,
        'inliers_pose': inliers_pose,
        'epi_err_mean': float(np.mean(epi_err)),
        'epi_err_median': float(np.median(epi_err)),
        'epi_err_p90': float(np.percentile(epi_err, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--pairs', type=int, default=30)
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.run_dir, 'images', '*.png')))
    assert len(imgs) > args.stride + 1, 'not enough images'

    # Sample pairs
    idxs = list(range(0, min(len(imgs)-args.stride, args.pairs*args.stride), args.stride))

    results = {k: [] for k in K_CANDIDATES.keys()}
    for i in idxs:
        im1 = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(imgs[i+args.stride], cv2.IMREAD_GRAYSCALE)
        if im1 is None or im2 is None:
            continue
        for name, K in K_CANDIDATES.items():
            r = evaluate_pair(im1, im2, K)
            if r:
                results[name].append(r)

    for name, arr in results.items():
        if not arr:
            print(name, 'no results')
            continue
        inE = [x['inliers_E'] for x in arr]
        inP = [x['inliers_pose'] for x in arr]
        em = [x['epi_err_mean'] for x in arr]
        emd = [x['epi_err_median'] for x in arr]
        e90 = [x['epi_err_p90'] for x in arr]
        print(f"\n{name}")
        print(f"  E inliers: mean={np.mean(inE):.1f} median={np.median(inE):.0f}")
        print(f"  Pose inliers: mean={np.mean(inP):.1f} median={np.median(inP):.0f}")
        print(f"  Epipolar error mean={np.mean(em):.3f} median={np.median(emd):.3f} p90={np.mean(e90):.3f}")

if __name__ == '__main__':
    main()
