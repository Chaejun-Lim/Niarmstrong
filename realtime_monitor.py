"""Print leader arm encoder angles and follower arm feedback in real time.

- Leader angles: raw serial samples from the Pico encoder board (see leader_interface.py),
  the same 7-value layout used by main2.py's JOINT_NAMES.
- Follower feedback: decoded state frames from the follower controller board
  (see follower_interface.py).

Usage:
    uv run python hybrid/realtime_monitor.py \
        --leader-port /dev/ttyACM0 \
        --follower-port /dev/ttyUSB0 \
        --hz 20
"""

from __future__ import annotations

import argparse
import time

from follower_interface import SerialFollowerArmController
from leader_interface import SerialLeaderArmInterface

JOINT_NAMES = [
    "joint_1_deg",
    "joint_2_deg",
    "joint_3_deg",
    "joint_4_deg",
    "joint_5_deg",
    "joint_6_deg",
    "gripper_deg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-baud", type=int, default=115200)
    parser.add_argument("--follower-port", default="/dev/ttyUSB0")
    parser.add_argument("--follower-baud", type=int, default=76800)
    parser.add_argument("--hz", type=float, default=20.0, help="Print rate")
    return parser.parse_args()


def _format_values(values) -> str:
    return " ".join(f"{name}={v:8.3f}" for name, v in zip(JOINT_NAMES, values))


def main() -> None:
    args = parse_args()
    leader = SerialLeaderArmInterface(args.leader_port, baudrate=args.leader_baud)
    follower = SerialFollowerArmController(args.follower_port, baudrate=args.follower_baud)

    period = 1.0 / args.hz if args.hz > 0 else 0.0
    try:
        while True:
            loop_start = time.perf_counter()

            leader_line = "leader: (연결 안 됨)"
            if leader.is_connected:
                try:
                    values, fresh = leader.read_action_sample()
                    tag = "new" if fresh else "held"
                    leader_line = f"leader[{tag}]: {_format_values(values)}"
                except RuntimeError:
                    leader_line = "leader: (샘플 없음)"

            follower_line = "follower: (연결 안 됨)"
            if follower.is_connected:
                values, fresh = follower.read_state_sample()
                tag = "new" if fresh else "held"
                follower_line = f"follower[{tag}]: {_format_values(values)}"

            print(f"{leader_line}\n{follower_line}\n")

            if period > 0:
                elapsed = time.perf_counter() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("[INFO] 중단됨 (Ctrl+C)")
    finally:
        leader.close()
        follower.close()


if __name__ == "__main__":
    main()
