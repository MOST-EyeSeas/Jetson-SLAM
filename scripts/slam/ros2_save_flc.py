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

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.max_frames = self.get_parameter('max_frames').get_parameter_value().integer_value

        self.images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.times_path = os.path.join(self.output_dir, 'times.txt')
        self.times_file = open(self.times_path, 'w')
        self.closed = False

        self.bridge = CvBridge()
        self.saved = 0
        self.subscription = self.create_subscription(Image, self.image_topic, self.cb, 10)
        self.get_logger().info(f"Saving images from {self.image_topic} to {self.images_dir}")

    def close_times(self):
        if not self.closed:
            try:
                self.times_file.flush()
                self.times_file.close()
            except Exception:
                pass
            self.closed = True

    def cb(self, msg: Image):
        # Use header stamp in nanoseconds for filename and times.txt
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        fname = os.path.join(self.images_dir, f"{stamp_ns}.png")
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv2.imwrite(fname, cv_img)
        self.times_file.write(f"{stamp_ns}\n")
        self.saved += 1
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
