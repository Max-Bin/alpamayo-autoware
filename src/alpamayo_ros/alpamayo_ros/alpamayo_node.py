#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""ROS 2 node that streams sensor topics into Alpamayo and publishes Autoware trajectories."""

from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import torch
from autoware_internal_debug_msgs.msg import StringStamped
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


class AlpamayoRosNode(Node):
    """ROS 2 node that consumes live topics (images + odometry) to run Alpamayo inference."""

    def __init__(self) -> None:
        super().__init__("alpamayo_node")

        self.model_name: str = "nvidia/Alpamayo-1.5-10B"
        self.declare_parameter("trajectory_topic", "/alpamayo/predicted_trajectory")
        self.declare_parameter("cot_topic", "/alpamayo/reasoning")
        self.declare_parameter("cot_with_stamped_topic", "/alpamayo/reasoning_stamped")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter("inference_period_sec", 0.1)

        # ROS2 Jazzy: Use non-empty default for string array parameters to properly infer type
        self.declare_parameter("camera_topics", [""])
        # Camera indices matching CAMERA_DISPLAY_NAMES in helper.py:
        # 0=Front left, 1=Front, 2=Front right, 3=Rear left, 4=Rear, 5=Rear right, 6=Front telephoto
        # Each index corresponds to the camera topic at the same position in camera_topics.
        self.declare_parameter("camera_indices", [0])

        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        KINEMATIC_STATE_HZ = 50.0
        ALPAMAYO_INPUT_HZ = 1.0
        self.skip_num = int(KINEMATIC_STATE_HZ / ALPAMAYO_INPUT_HZ)

        self._num_history_steps = 16
        self._num_frames = 4
        inference_period = float(self.get_parameter("inference_period_sec").value)

        queue_size = 10
        traj_topic = self.get_parameter("trajectory_topic").value
        self._trajectory_pub = self.create_publisher(Trajectory, traj_topic, queue_size)
        self.get_logger().info(f"Publishing Autoware trajectories on {traj_topic}")

        cot_topic = self.get_parameter("cot_topic").value
        self._cot_pub = self.create_publisher(String, cot_topic, queue_size)
        self.get_logger().info(f"Publishing reasoning traces on {cot_topic}")

        cot_stamped_topic = self.get_parameter("cot_with_stamped_topic").value
        self._cot_stamped_pub = self.create_publisher(StringStamped, cot_stamped_topic, queue_size)
        self.get_logger().info(f"Publishing reasoning traces (stamped) on {cot_stamped_topic}")

        marker_topic = traj_topic + "_markers"
        self._marker_pub = self.create_publisher(MarkerArray, marker_topic, queue_size)
        self.get_logger().info(f"Publishing trajectory markers on {marker_topic}")

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._active_future: Optional[Future] = None

        self._frame_id = "base_link"

        camera_topics = list(
            self.get_parameter("camera_topics").get_parameter_value().string_array_value
        )
        if not camera_topics:
            raise ValueError("camera_topics parameter must list at least one image topic.")
        self._camera_topics = camera_topics

        camera_indices = list(
            self.get_parameter("camera_indices").get_parameter_value().integer_array_value
        )
        if len(camera_indices) != len(camera_topics):
            raise ValueError(
                f"camera_indices length ({len(camera_indices)}) must match "
                f"camera_topics length ({len(camera_topics)})."
            )
        self._camera_indices = torch.tensor(camera_indices, dtype=torch.int64)
        self._camera_buffers: Dict[str, deque] = {
            topic: deque(maxlen=self._num_frames * 3) for topic in self._camera_topics
        }

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10
        )

        for topic in self._camera_topics:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, t=topic: self._image_callback(t, msg),
                camera_qos,
            )
            self.get_logger().info(f"Subscribed to camera topic: {topic}")

        self._odometry_buffer: deque[Odometry] = deque(
            maxlen=self._num_history_steps * self.skip_num + 10
        )
        odom_topic = self.get_parameter("odometry_topic").value

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50
        )
        self.create_subscription(Odometry, odom_topic, self._odometry_callback, odom_qos)
        self.get_logger().info(f"Subscribed to odometry topic: {odom_topic}")

        self._auto_timer = self.create_timer(inference_period, self._timer_callback)
        self.get_logger().info(
            f"Loading Alpamayo model {self.model_name} on device={self._device} dtype={self._dtype}"
        )
        self._model = Alpamayo1_5.from_pretrained(self.model_name, dtype=self._dtype).to(
            self._device
        )
        self._model.eval()
        self._processor = helper.get_processor(self._model.tokenizer)

        # Set random seed once during initialization
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.get_logger().info("Alpamayo model loaded and ready.")

    def destroy_node(self) -> None:
        """Cleanup resources before shutting down."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()

    def _timer_callback(self) -> None:
        if self._active_future and not self._active_future.done():
            return
        payload = self._prepare_inference_payload()
        if payload is None:
            return
        self.get_logger().info("Starting Alpamayo inference from streaming data.")
        self._active_future = self._executor.submit(self._run_inference, payload)
        self._active_future.add_done_callback(self._on_future_done)

    def _image_callback(self, topic: str, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warning("Failed to decode compressed image.")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        self._camera_buffers[topic].append((msg.header.stamp, tensor))

    def _odometry_callback(self, msg: Odometry) -> None:
        self._odometry_buffer.append(msg)

    def _prepare_inference_payload(self) -> Optional[dict]:
        if not all(len(buf) >= self._num_frames for buf in self._camera_buffers.values()):
            return None
        camera_tensors: List[torch.Tensor] = []
        for topic in self._camera_topics:
            frames = list(self._camera_buffers[topic])[-self._num_frames :]
            tensors = [frame for _, frame in frames]
            camera_tensors.append(torch.stack(tensors, dim=0))
        image_frames = torch.stack(camera_tensors, dim=0)

        if len(self._odometry_buffer) < self._num_history_steps * self.skip_num:
            return None
        odom_history = list(self._odometry_buffer)[
            -self._num_history_steps * self.skip_num :: self.skip_num
        ]

        # Build history tensors
        positions = []
        rotations = []
        for msg in odom_history:
            pose = msg.pose.pose
            positions.append([pose.position.x, pose.position.y, pose.position.z])
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rotations.append(Rotation.from_quat(quat).as_matrix())
        positions_np = np.asarray(positions, dtype=np.float32)
        rotations_np = np.asarray(rotations, dtype=np.float32)
        t0_rot_inv = np.linalg.inv(rotations_np[-1])
        centered = positions_np - positions_np[-1]
        history_xyz_local = centered @ t0_rot_inv.T
        history_rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rotations_np)
        ego_history_xyz = torch.from_numpy(history_xyz_local).unsqueeze(0).unsqueeze(0)
        ego_history_rot = torch.from_numpy(history_rot_local).unsqueeze(0).unsqueeze(0)

        return {
            "image_frames": image_frames,
            "camera_indices": self._camera_indices,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }

    def _run_inference(self, payload: dict) -> dict:
        start = time.time()
        frames = payload["image_frames"]
        messages = helper.create_message(
            frames.flatten(0, 1),
            camera_indices=payload["camera_indices"],
            num_frames_per_camera=self._num_frames,
        )
        processor_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": processor_inputs,
            "ego_history_xyz": payload["ego_history_xyz"],
            "ego_history_rot": payload["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, device=self._device)

        generation_kwargs = {
            "top_p": 0.98,
            "temperature": 0.6,
            "num_traj_samples": 1,
            "num_traj_sets": 1,
            "max_generation_length": 64,
            "return_extra": True,
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self._model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                **generation_kwargs,
            )

        pred_xyz_cpu = pred_xyz.detach().cpu()
        pred_rot_cpu = pred_rot.detach().cpu()
        trajectory = pred_xyz_cpu[0, 0, 0]
        rotation = pred_rot_cpu[0, 0, 0]
        traj_msg = self._to_autoware_trajectory(trajectory, rotation)
        self._trajectory_pub.publish(traj_msg)

        marker_array = self._trajectory_to_markers(trajectory)
        self._marker_pub.publish(marker_array)

        cot_text = self._extract_text(extra, "cot")
        if cot_text:
            cot_msg = String()
            cot_msg.data = cot_text
            self._cot_pub.publish(cot_msg)

            cot_stamped_msg = StringStamped()
            cot_stamped_msg.stamp = traj_msg.header.stamp
            cot_stamped_msg.data = cot_text
            self._cot_stamped_pub.publish(cot_stamped_msg)

        duration = time.time() - start
        return {"duration_sec": duration, "num_poses": len(traj_msg.points)}

    def _to_autoware_trajectory(
        self,
        trajectory: torch.Tensor,
        rotations: torch.Tensor | None,
    ) -> Trajectory:
        traj_np = trajectory.numpy()
        rot_np = rotations.numpy() if rotations is not None else None
        now = self.get_clock().now().to_msg()

        traj_msg = Trajectory()
        traj_msg.header.stamp = now
        traj_msg.header.frame_id = self._frame_id

        dt = 0.1
        prev_xy = None

        for idx, point in enumerate(traj_np):
            traj_point = TrajectoryPoint()
            traj_point.pose.position.x = float(point[0])
            traj_point.pose.position.y = float(point[1])
            traj_point.pose.position.z = float(point[2])

            if rot_np is not None:
                quat = Rotation.from_matrix(rot_np[idx]).as_quat()
                traj_point.pose.orientation.x = float(quat[0])
                traj_point.pose.orientation.y = float(quat[1])
                traj_point.pose.orientation.z = float(quat[2])
                traj_point.pose.orientation.w = float(quat[3])
            else:
                traj_point.pose.orientation.w = 1.0

            if prev_xy is None:
                speed = 0.0
            else:
                dx = float(point[0] - prev_xy[0])
                dy = float(point[1] - prev_xy[1])
                dist = math.hypot(dx, dy)
                speed = dist / dt if dt > 0 else 0.0

            traj_point.longitudinal_velocity_mps = float(speed)
            traj_point.lateral_velocity_mps = 0.0
            traj_point.acceleration_mps2 = 0.0
            traj_point.heading_rate_rps = 0.0

            seconds_float = idx * dt
            seconds_int = int(seconds_float)
            nanosec = int((seconds_float - seconds_int) * 1e9)
            traj_point.time_from_start = Duration(sec=seconds_int, nanosec=nanosec)

            traj_msg.points.append(traj_point)
            prev_xy = (point[0], point[1])

        return traj_msg

    def _trajectory_to_markers(self, trajectory: torch.Tensor) -> MarkerArray:
        traj_np = trajectory.numpy()
        now = self.get_clock().now().to_msg()

        marker_array = MarkerArray()

        # LINE_STRIP marker for trajectory path
        line_marker = Marker()
        line_marker.header.stamp = now
        line_marker.header.frame_id = self._frame_id
        line_marker.ns = "trajectory"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 1.0  # Line width
        line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green
        line_marker.pose.orientation.w = 1.0

        for point in traj_np:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = float(point[2])
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)
        return marker_array

    def _extract_text(self, extra: dict, key: str) -> Optional[str]:
        if not extra or key not in extra:
            return None
        text_array = extra[key]
        try:
            text = text_array[0, 0, 0]
        except Exception:
            return None
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        text = str(text).strip()
        return text or None

    def _on_future_done(self, future: Future) -> None:
        try:
            metrics = future.result()
        except Exception as exc:
            self.get_logger().error(f"Alpamayo inference failed: {exc}")
            return
        if not metrics:
            return
        self.get_logger().info(
            f"Alpamayo inference completed in {metrics['duration_sec']:.2f}s "
            f"(points={metrics['num_poses']})."
        )


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AlpamayoRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
