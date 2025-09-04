#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import numpy as np

def load_gt_csv(path: Path):
    ts=[]; xyz=[]
    with path.open() as f:
        r=csv.DictReader(f)
        for row in r:
            ts.append(int(row['t_ns']))
            xyz.append([float(row['x']),float(row['y']),float(row['z'])])
    return np.array(ts, dtype=np.int64), np.array(xyz, dtype=np.float64)

def load_slam_kf(path: Path):
    ts=[]; xyz=[]
    with path.open() as f:
        for ln in f:
            parts=ln.split()
            if len(parts)==8:
                t_ns=int(float(parts[0])*1e9)
                x,y,z=map(float, parts[1:4])
                ts.append(t_ns); xyz.append([x,y,z])
    return np.array(ts, dtype=np.int64), np.array(xyz, dtype=np.float64)

def _nearest_indices(ts_ref: np.ndarray, ts_query: np.ndarray) -> np.ndarray:
    """
    For each timestamp in ts_query, pick the nearest index in ts_ref.
    Both arrays are int64 nanoseconds, assumed sorted ascending.
    Returns an array of indices into ts_ref, same length as ts_query.
    """
    idx = np.searchsorted(ts_ref, ts_query, side='left')
    idx = np.clip(idx, 0, len(ts_ref) - 1)
    # choose closer of left/right neighbor
    left = np.maximum(idx - 1, 0)
    right = idx
    left_diff = np.abs(ts_ref[left] - ts_query)
    right_diff = np.abs(ts_ref[right] - ts_query)
    choose_left = left_diff < right_diff
    out = np.where(choose_left, left, right)
    return out


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Umeyama 3D similarity alignment. Solves for s, R, t that minimizes
    || dst - (s * R @ src + t) ||_F. Returns s (float), R (3x3), t (3,).
    src, dst: (N,3)
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    X = src - mean_src
    Y = dst - mean_dst
    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = np.sum(X * X) / n
    if with_scale:
        scale = np.trace(np.diag(D) @ S) / var_src
    else:
        scale = 1.0
    t = mean_dst - scale * (R @ mean_src)
    return float(scale), R, t


def _compute_rpe(xyz_aligned: np.ndarray, xyz_gt_matched: np.ndarray, delta: int = 1) -> float:
    """
    Relative Pose Error (translational) RMSE over step size delta (indices).
    """
    if len(xyz_aligned) <= delta:
        return float('nan')
    d_est = xyz_aligned[delta:] - xyz_aligned[:-delta]
    d_gt = xyz_gt_matched[delta:] - xyz_gt_matched[:-delta]
    de = d_est - d_gt
    se = np.sum(de ** 2, axis=1)
    return math.sqrt(float(np.mean(se)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_csv', required=True, help='CSV from ground_truth_logger.py')
    ap.add_argument('--kf_txt', default='/home/Jetson-SLAM/Examples/Monocular/KeyFrameTrajectory.txt', help='SLAM keyframe trajectory file')
    args = ap.parse_args()

    ts_gt, xyz_gt = load_gt_csv(Path(args.gt_csv))
    ts_kf, xyz_kf = load_slam_kf(Path(args.kf_txt))
    if len(ts_gt)==0 or len(ts_kf)==0:
        print('No data. gt:',len(ts_gt),' kf:',len(ts_kf)); return

    # Time match: nearest GT per SLAM KF
    idx = _nearest_indices(ts_gt, ts_kf)
    xyz_gt_matched = xyz_gt[idx]

    # Align (Sim(3)) SLAM to GT
    try:
        s, R, t = _umeyama_similarity(xyz_kf, xyz_gt_matched, with_scale=True)
        xyz_kf_aligned = (s * (xyz_kf @ R.T)) + t
    except Exception:
        # Fallback: no alignment
        xyz_kf_aligned = xyz_kf.copy()

    err = xyz_kf_aligned - xyz_gt_matched
    se = np.sum(err**2, axis=1)
    rmse = math.sqrt(float(np.mean(se)))
    p95 = math.sqrt(float(np.percentile(se, 95)))

    # RPE (delta=1)
    rpe = _compute_rpe(xyz_kf_aligned, xyz_gt_matched, delta=1)
    # Print an auxiliary line first (sweeper uses last line)
    print('Aligned RPE_trans(m)@delta=1:', rpe)
    print('Keyframes:',len(ts_kf),' RMSE_pos(m):',rmse,' P95(m):',p95)

if __name__ == '__main__':
    main()
