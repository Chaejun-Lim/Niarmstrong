"""Run inference with a trained pi0-style policy checkpoint.

Input state can come from:
- --state-csv "v1,v2,v3,v4,v5,v6,v7"
- --state-npy path/to/state.npy (uses the last row)
- --control-shm /hybrid_control (reads latest follower state from shared memory)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pi0_policy import PiZeroConfig, PiZeroPolicy, denormalize, normalize
from shm_layout import ControlSharedBlock, MappedStruct


class ControlSharedReader:
    def __init__(self, shm_name: str):
        self._mapped = MappedStruct(shm_name, ControlSharedBlock)
        self.block = self._mapped.view

    def get_latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        action = None
        state = None
        if self.block.cmd.valid:
            action = np.array(self.block.cmd.action_deg, dtype=np.float32)
        if self.block.state.valid:
            state = np.array(self.block.state.state_deg, dtype=np.float32)
        return action, state

    def close(self) -> None:
        self._mapped.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-csv", default="", help="7 comma-separated joint degrees")
    parser.add_argument("--state-npy", type=Path, default=None)
    parser.add_argument("--control-shm", default="", help="Read latest state from control shared memory")
    parser.add_argument("--state-fallback-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    selected = int(bool(args.state_csv)) + int(args.state_npy is not None) + int(bool(args.control_shm))
    if selected != 1:
        parser.error("Choose exactly one source: --state-csv, --state-npy, or --control-shm")
    return args


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def parse_state_csv(raw: str) -> np.ndarray:
    values = np.fromstring(raw, sep=",", dtype=np.float32)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError("--state-csv must contain exactly 7 finite numbers")
    return values


def parse_state_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 1:
        state = arr.astype(np.float32, copy=False)
    elif arr.ndim == 2:
        state = arr[-1].astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"Unsupported npy rank: {arr.ndim}")
    if state.shape != (7,) or not np.isfinite(state).all():
        raise RuntimeError("--state-npy must contain state with shape (7,) or (N,7)")
    return state


def read_state_from_shm(shm_name: str, timeout_sec: float, state_fallback_action: bool) -> np.ndarray:
    reader = ControlSharedReader(shm_name)
    deadline = time.perf_counter() + timeout_sec
    try:
        while time.perf_counter() < deadline:
            action, state = reader.get_latest()
            if state is not None:
                return state
            if state_fallback_action and action is not None:
                return action
            time.sleep(0.01)
    finally:
        reader.close()

    raise RuntimeError(f"Timed out waiting for state from shared memory: {shm_name}")


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[PiZeroPolicy, dict[str, torch.Tensor]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = PiZeroConfig(**ckpt["config"])
    model = PiZeroPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = {
        "state_mean": ckpt["state_mean"].to(device),
        "state_std": ckpt["state_std"].to(device),
        "action_mean": ckpt["action_mean"].to(device),
        "action_std": ckpt["action_std"].to(device),
    }
    return model, stats


@torch.no_grad()
def predict_action(model: PiZeroPolicy, stats: dict[str, torch.Tensor], state: np.ndarray, device: torch.device) -> np.ndarray:
    state_t = torch.from_numpy(state.astype(np.float32, copy=False)).to(device).unsqueeze(0)
    state_n = normalize(state_t, stats["state_mean"], stats["state_std"])
    pred_n = model(state_n)
    pred = denormalize(pred_n, stats["action_mean"], stats["action_std"])
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, stats = load_policy(args.checkpoint, device)

    if args.state_csv:
        state = parse_state_csv(args.state_csv)
        source = "state_csv"
    elif args.state_npy is not None:
        state = parse_state_npy(args.state_npy)
        source = "state_npy"
    else:
        state = read_state_from_shm(args.control_shm, args.timeout_sec, args.state_fallback_action)
        source = "control_shm"

    action = predict_action(model, stats, state, device)
    action_csv = ",".join(f"{x:.6f}" for x in action.tolist())
    print(f"[INFO] source={source} device={device}")
    print(f"[INFO] state_deg={','.join(f'{x:.6f}' for x in state.tolist())}")
    print(f"[INFO] action_deg={action_csv}")

    if args.json_out is not None:
        payload = {
            "source": source,
            "state_deg": [float(x) for x in state.tolist()],
            "action_deg": [float(x) for x in action.tolist()],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] wrote json: {args.json_out}")


if __name__ == "__main__":
    main()
