#!/usr/bin/env python3
import argparse
import csv
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped

class GTLogger(Node):
    def __init__(self, out_path: str, max_msgs: int, duration_sec: float):
        super().__init__('gt_logger')
        self._f = open(out_path, 'w', newline='')
        self._w = csv.writer(self._f)
        self._w.writerow(['t_ns', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self._n = 0
        self._max = max_msgs
        self._duration = duration_sec
        if duration_sec > 0:
            self._deadline = self.get_clock().now() + Duration(seconds=self._duration)
            self._timer = self.create_timer(0.2, self._check_deadline)
        self.create_subscription(PoseStamped, '/ground_truth/pose', self._cb, 10)

    def _check_deadline(self):
        if self.get_clock().now() >= self._deadline:
            self._shutdown()

    def _cb(self, msg: PoseStamped):
        t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        p = msg.pose.position
        q = msg.pose.orientation
        self._w.writerow([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        self._n += 1
        if self._max > 0 and self._n >= self._max:
            self._shutdown()

    def _shutdown(self):
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='CSV output path')
    ap.add_argument('--max_msgs', type=int, default=0, help='Max messages before exit (0=unbounded)')
    ap.add_argument('--duration_sec', type=float, default=0.0, help='Max duration in seconds (0=unbounded)')
    args = ap.parse_args()

    rclpy.init()
    node = GTLogger(args.out, args.max_msgs, args.duration_sec)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._f.close()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
