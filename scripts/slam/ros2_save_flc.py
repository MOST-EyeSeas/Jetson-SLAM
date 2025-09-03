#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FLCImageSaver(Node):
    def __init__(self):
        super().__init__('flc_image_saver')
        self.declare_parameter('image_topic', '/camera/flc/image_raw')
        self.declare_parameter('output_dir', os.path.expanduser('~/Jetson-SLAM/data/flc_run'))
        self.declare_parameter('max_frames', 0)  # 0 = unlimited
        # Preprocessing options
        self.declare_parameter('use_grayscale', True)
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('use_sharpen', False)
        # Timestamp handling
        self.declare_parameter('timestamp_source', 'header')  # 'header' or 'arrival'
        self.declare_parameter('enforce_monotonic', True)
        self.declare_parameter('monotonic_strategy', 'clamp')  # 'clamp' or 'skip'

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.max_frames = self.get_parameter('max_frames').get_parameter_value().integer_value
        self.use_grayscale = self.get_parameter('use_grayscale').get_parameter_value().bool_value
        self.use_clahe = self.get_parameter('use_clahe').get_parameter_value().bool_value
        self.use_sharpen = self.get_parameter('use_sharpen').get_parameter_value().bool_value
        self.timestamp_source = self.get_parameter('timestamp_source').get_parameter_value().string_value
        self.enforce_monotonic = self.get_parameter('enforce_monotonic').get_parameter_value().bool_value
        self.monotonic_strategy = self.get_parameter('monotonic_strategy').get_parameter_value().string_value

        self.images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.times_path = os.path.join(self.output_dir, 'times.txt')
        self.times_file = open(self.times_path, 'w')
        self.closed = False

        self.bridge = CvBridge()
        self.saved = 0
        self.last_stamp_ns = -1
        self.subscription = self.create_subscription(Image, self.image_topic, self.cb, 10)
        self.get_logger().info(f"Saving images from {self.image_topic} to {self.images_dir}")
        self.get_logger().info(f"Preprocess: grayscale={self.use_grayscale}, clahe={self.use_clahe}, sharpen={self.use_sharpen}")
        self.get_logger().info(
            f"Timestamps: source={self.timestamp_source}, enforce_monotonic={self.enforce_monotonic}, strategy={self.monotonic_strategy}"
        )

    def close_times(self):
        if not self.closed:
            try:
                self.times_file.flush()
                self.times_file.close()
            except Exception:
                pass
            self.closed = True

    def preprocess(self, img):
        out = img
        if self.use_grayscale:
            if len(out.shape) == 3:
                out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            if len(out.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                out = clahe.apply(out)
            else:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l2 = clahe.apply(l)
                out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
        if self.use_sharpen:
            # Mild unsharp mask
            blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
            out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
        return out

    def cb(self, msg: Image):
        # Choose timestamp source
        if self.timestamp_source == 'arrival':
            now = self.get_clock().now()
            stamp_ns = int(now.nanoseconds)
        else:
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Enforce strict monotonic increase if requested
        if self.enforce_monotonic and self.last_stamp_ns >= 0 and stamp_ns <= self.last_stamp_ns:
            if self.monotonic_strategy == 'skip':
                # Skip non-monotonic frame
                return
            # Clamp to last + 1 ns to ensure uniqueness and order
            stamp_ns = self.last_stamp_ns + 1
        fname = os.path.join(self.images_dir, f"{stamp_ns}.png")
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_img = self.preprocess(cv_img)
        cv2.imwrite(fname, cv_img)
        self.times_file.write(f"{stamp_ns}\n")
        self.saved += 1
        self.last_stamp_ns = stamp_ns
        if self.saved % 50 == 0:
            self.get_logger().info(f"Saved {self.saved} frames")
        if self.max_frames > 0 and self.saved >= self.max_frames:
            self.get_logger().info("Reached max_frames; shutting down")
            self.close_times()
            if rclpy.ok():
                rclpy.shutdown()


def main():
    rclpy.init()
    node = FLCImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_times()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
