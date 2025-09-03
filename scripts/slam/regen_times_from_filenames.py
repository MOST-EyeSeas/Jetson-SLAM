#!/usr/bin/env python3
import os
import glob
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='Recording root (contains images/)')
    ap.add_argument('--output', default='times_sorted.txt')
    args = ap.parse_args()

    img_dir = os.path.join(args.run_dir, 'images')
    files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if not files:
        print('No images found in', img_dir)
        return 1
    out_path = os.path.join(args.run_dir, args.output)
    with open(out_path, 'w') as f:
        for p in files:
            ts = os.path.splitext(os.path.basename(p))[0]
            f.write(ts + '\n')
    print('Wrote', out_path, 'with', len(files), 'timestamps')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
