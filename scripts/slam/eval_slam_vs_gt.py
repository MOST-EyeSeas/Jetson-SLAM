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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_csv', required=True, help='CSV from ground_truth_logger.py')
    ap.add_argument('--kf_txt', default='/home/Jetson-SLAM/Examples/Monocular/KeyFrameTrajectory.txt', help='SLAM keyframe trajectory file')
    args = ap.parse_args()

    ts_gt, xyz_gt = load_gt_csv(Path(args.gt_csv))
    ts_kf, xyz_kf = load_slam_kf(Path(args.kf_txt))
    if len(ts_gt)==0 or len(ts_kf)==0:
        print('No data. gt:',len(ts_gt),' kf:',len(ts_kf)); return

    idx = np.searchsorted(ts_gt, ts_kf, side='left')
    idx = np.clip(idx, 0, len(ts_gt)-1)
    err = xyz_kf - xyz_gt[idx]
    se = np.sum(err**2, axis=1)
    rmse = math.sqrt(float(np.mean(se)))
    p95 = math.sqrt(float(np.percentile(se, 95)))
    print('Keyframes:',len(ts_kf),' RMSE_pos(m):',rmse,' P95(m):',p95)

if __name__ == '__main__':
    main()
