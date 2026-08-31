"""Train a robot-agnostic pi0-style BC policy from hybrid/main.py recordings.

This script does not require a LeRobot robot integration. It reads local episode
folders containing state.npy and action.npy, then trains a policy for
observation.state -> action imitation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from pi0_policy import PiZeroConfig, PiZeroPolicy, normalize


class StateActionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("states/actions must be rank-2 arrays")
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions must have the same number of rows")
        self.states = torch.from_numpy(states.astype(np.float32, copy=False))
        self.actions = torch.from_numpy(actions.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path like hybrid/recordings/<account>/<task>",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pi0_runs/run1"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means unlimited")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be >= 1")
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("val-ratio must be in [0, 1)")
    return args


def find_episode_dirs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")
    episodes = []
    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "state.npy").is_file() and (child / "action.npy").is_file():
            episodes.append(child)
    if not episodes:
        raise RuntimeError(f"No episodes with state.npy/action.npy found under: {dataset_dir}")
    return episodes


def load_arrays(episodes: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    for ep in episodes:
        states = np.load(ep / "state.npy")
        actions = np.load(ep / "action.npy")
        if states.ndim != 2 or actions.ndim != 2:
            raise RuntimeError(f"Invalid shapes in {ep}: {states.shape}, {actions.shape}")
        if states.shape != actions.shape:
            raise RuntimeError(f"State/action shape mismatch in {ep}: {states.shape} vs {actions.shape}")
        if states.shape[1] != 7:
            raise RuntimeError(f"Expected 7 joints in {ep}, got {states.shape[1]}")
        all_states.append(states.astype(np.float32, copy=False))
        all_actions.append(actions.astype(np.float32, copy=False))

    states_cat = np.concatenate(all_states, axis=0)
    actions_cat = np.concatenate(all_actions, axis=0)
    if not np.isfinite(states_cat).all() or not np.isfinite(actions_cat).all():
        raise RuntimeError("Dataset contains NaN/Inf values")
    return states_cat, actions_cat


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.SmoothL1Loss()
    total = 0.0
    count = 0
    for s, a in loader:
        s = s.to(device)
        a = a.to(device)
        s_n = normalize(s, state_mean, state_std)
        a_n = normalize(a, action_mean, action_std)
        pred_n = model(s_n)
        loss = criterion(pred_n, a_n)
        total += float(loss.item()) * s.shape[0]
        count += int(s.shape[0])
    return total / max(1, count)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    episodes = find_episode_dirs(args.dataset_dir)
    states, actions = load_arrays(episodes)

    config = PiZeroConfig(
        state_dim=int(states.shape[1]),
        action_dim=int(actions.shape[1]),
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        dropout=args.dropout,
    )

    dataset = StateActionDataset(states, actions)
    n_total = len(dataset)
    n_val = int(math.floor(n_total * args.val_ratio))
    n_train = n_total - n_val
    if n_train <= 0:
        raise RuntimeError("Train split is empty. Reduce --val-ratio or collect more data.")

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = resolve_device(args.device)
    model = PiZeroPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    train_states = dataset.states[train_set.indices]
    train_actions = dataset.actions[train_set.indices]
    state_mean = train_states.mean(dim=0, keepdim=True).to(device)
    state_std = train_states.std(dim=0, keepdim=True).clamp_min(1e-4).to(device)
    action_mean = train_actions.mean(dim=0, keepdim=True).to(device)
    action_std = train_actions.std(dim=0, keepdim=True).clamp_min(1e-4).to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "policy_best.pt"
    latest_path = args.output_dir / "policy_latest.pt"
    metrics_path = args.output_dir / "train_metrics.json"

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    global_step = 0
    start_time = time.time()

    print(f"[INFO] episodes={len(episodes)} samples={n_total} train={n_train} val={n_val}")
    print(f"[INFO] device={device} amp={'on' if scaler.is_enabled() else 'off'}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for s, a in train_loader:
            s = s.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)

            s_n = normalize(s, state_mean, state_std)
            a_n = normalize(a, action_mean, action_std)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred_n = model(s_n)
                loss = criterion(pred_n, a_n)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item()) * s.shape[0]
            train_count += int(s.shape[0])
            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = evaluate(model, val_loader, state_mean, state_std, action_mean, action_std, device) if n_val > 0 else train_loss
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "global_step": global_step,
        }
        history.append(row)
        print(
            f"[E{epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} "
            f"lr={row['lr']:.2e} step={global_step}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "state_mean": state_mean.detach().cpu(),
            "state_std": state_std.detach().cpu(),
            "action_mean": action_mean.detach().cpu(),
            "action_std": action_std.detach().cpu(),
            "history": history,
        }
        torch.save(ckpt, latest_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_path)
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, args.output_dir / f"policy_epoch_{epoch:03d}.pt")

        if args.max_steps > 0 and global_step >= args.max_steps:
            print("[INFO] reached --max-steps, ending training early")
            break

    elapsed = time.time() - start_time
    metrics = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "episodes": len(episodes),
        "samples": n_total,
        "train_samples": n_train,
        "val_samples": n_val,
        "best_val_loss": best_val,
        "elapsed_sec": elapsed,
        "config": config.to_dict(),
        "args": _json_safe(vars(args)),
        "history": history,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] training completed")
    print(f"[INFO] best checkpoint: {best_path}")
    print(f"[INFO] latest checkpoint: {latest_path}")
    print(f"[INFO] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
