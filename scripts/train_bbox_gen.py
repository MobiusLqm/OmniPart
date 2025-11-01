"""Training script for the BboxGen autoregressive bounding-box generator.

This script mirrors the data preparation steps that are executed during
inference in :mod:`scripts.inference_omnipart` while adding the components that
are required for teacher-forced training of :class:`modules.bbox_gen.models
.autogressive_bbox_gen.BboxGen`.

The script expects a metadata file describing the assets required for each
training sample.  Each entry must contain four fields: ``image`` (RGBA render),
``mask`` (segmentation map aligned with the render), ``voxel`` (voxel
coordinates saved with ``numpy.save``) and ``bbox`` (bounding boxes saved with
``numpy.save``).  Paths can be absolute or relative to ``--dataset_root``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from omegaconf import OmegaConf

from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from modules.bbox_gen.utils.bbox_tokenizer import BoundsTokenizerDiag
from modules.inference_utils import change_pcd_range, load_img_mask


@dataclass
class SamplePaths:
    """Container for resolved file paths associated with a sample."""

    image: str
    mask: str
    voxel: str
    bbox: str


def load_metadata(path: str) -> List[Dict[str, Any]]:
    """Load dataset metadata from ``path``.

    Supports JSON and JSONL formats.
    """

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def resolve_path(root: Optional[str], file_path: str) -> str:
    """Resolve ``file_path`` relative to ``root`` if the path is not absolute."""

    if os.path.isabs(file_path) or root is None:
        return file_path
    return os.path.join(root, file_path)


class BBoxGenDataset(Dataset):
    """Dataset that prepares inputs for ``BboxGen`` training."""

    def __init__(
        self,
        metadata: Iterable[Dict[str, Any]],
        cfg,
        dataset_root: Optional[str] = None,
        coord_range: Tuple[float, float] = (-0.5, 0.5),
    ) -> None:
        super().__init__()
        self.samples: List[SamplePaths] = [
            SamplePaths(
                image=resolve_path(dataset_root, entry["image"]),
                mask=resolve_path(dataset_root, entry["mask"]),
                voxel=resolve_path(dataset_root, entry["voxel"]),
                bbox=resolve_path(dataset_root, entry["bbox"]),
            )
            for entry in metadata
        ]

        self.cfg = cfg
        self.coord_range = coord_range
        self.tokenizer = BoundsTokenizerDiag(
            bins=self.cfg.bins,
            BOS_id=self.cfg.BOS_id,
            EOS_id=self.cfg.EOS_id,
            PAD_id=self.cfg.PAD_id,
        )

    def __len__(self) -> int:  # pragma: no cover - simple container method
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        img_white_bg, _, ordered_mask_input, _ = load_img_mask(sample.image, sample.mask)

        whole_voxel = np.load(sample.voxel, allow_pickle=False)
        if whole_voxel.ndim != 2 or whole_voxel.shape[1] not in (3, 4):
            raise ValueError(f"Unexpected voxel coordinate shape: {whole_voxel.shape}")

        if whole_voxel.shape[1] == 4:
            whole_voxel = whole_voxel[:, 1:4]
        whole_voxel = whole_voxel.astype(np.float32)
        whole_voxel = (whole_voxel + 0.5) / self.cfg.bins - 0.5
        whole_voxel_index = change_pcd_range(
            whole_voxel, from_rg=(-0.5, 0.5), to_rg=(0.5 / self.cfg.bins, 1 - 0.5 / self.cfg.bins)
        )
        whole_voxel_index = (whole_voxel_index * self.cfg.bins).astype(np.int32)

        bounds = np.load(sample.bbox, allow_pickle=False).astype(np.float32)
        tokens = self.tokenizer.encode({"bounds": bounds}, coord_rg=self.coord_range)
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        input_ids = torch.full((self.cfg.max_length,), self.cfg.PAD_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        voxel_token_length = self.cfg.voxel_token_length
        input_ids[:voxel_token_length] = self.cfg.voxel_token_placeholder

        if token_tensor.numel() > self.cfg.max_length - voxel_token_length:
            raise ValueError(
                "Tokenized bounding boxes exceed maximum sequence length. "
                f"Got {token_tensor.numel()} tokens but only {self.cfg.max_length - voxel_token_length} slots are available."
            )

        token_end = voxel_token_length + token_tensor.numel()
        input_ids[voxel_token_length:token_end] = token_tensor
        labels[voxel_token_length:token_end] = token_tensor

        return {
            "images": img_white_bg.float(),
            "masks": ordered_mask_input.long(),
            "points": torch.from_numpy(whole_voxel).float(),
            "whole_voxel_index": torch.from_numpy(whole_voxel_index).long(),
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_batch(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack tensors from ``samples`` into a mini-batch."""

    batch = {}
    for key in samples[0].keys():
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    return batch


def setup_distributed() -> Tuple[int, int, int]:
    """Initialise distributed training if environment variables are set."""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():  # pragma: no cover - infrastructure
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Create a cosine scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: str,
    epoch: int,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: Optional[GradScaler],
    is_main_process: bool,
) -> None:
    if not is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch{epoch:04d}_step{step:08d}.pt"))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int,
    log_interval: int,
    is_main_process: bool,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, disable=not is_main_process, desc=f"Epoch {epoch}")

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress, start=1):
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=scaler.is_enabled()):
            outputs = model(batch)
            logits = outputs["logits"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if step % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.detach().float().item() * gradient_accumulation_steps
        num_batches += 1

        if is_main_process and step % log_interval == 0:
            progress.set_postfix({"loss": total_loss / num_batches})

    # Flush gradients if the number of steps was not divisible by the accumulation factor
    if len(dataloader) % gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_main_process: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, disable=not is_main_process, desc="Validation")
    for batch in progress:
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = labels.to(device, non_blocking=True)

        outputs = model(batch)
        logits = outputs["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total_loss += loss.detach().float().item()
        num_batches += 1
        if is_main_process:
            progress.set_postfix({"loss": total_loss / num_batches})

    return total_loss / max(1, num_batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BboxGen model")
    parser.add_argument("--train_metadata", type=str, required=True, help="Path to JSON/JSONL metadata for the training split")
    parser.add_argument("--val_metadata", type=str, default=None, help="Path to JSON/JSONL metadata for the validation split")
    parser.add_argument("--dataset_root", type=str, default=None, help="Optional root that prefixes all relative asset paths")
    parser.add_argument("--config", type=str, default="configs/bbox_gen.yaml", help="Path to the bbox generator config")
    parser.add_argument("--partfield_encoder_path", type=str, required=True, help="Checkpoint of the frozen PartField encoder")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store checkpoints")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    set_seed(args.seed + rank)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    bbox_gen_config = OmegaConf.load(args.config).model.args
    bbox_gen_config.partfield_encoder_path = args.partfield_encoder_path

    train_metadata = load_metadata(args.train_metadata)
    train_dataset = BBoxGenDataset(train_metadata, bbox_gen_config, dataset_root=args.dataset_root)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    if args.val_metadata is not None:
        val_metadata = load_metadata(args.val_metadata)
        val_dataset = BBoxGenDataset(val_metadata, bbox_gen_config, dataset_root=args.dataset_root)
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if world_size > 1
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )
    else:
        val_loader = None

    model = BboxGen(bbox_gen_config)
    model.to(device)

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = (len(train_loader) * args.num_epochs) // max(1, args.gradient_accumulation_steps)
    scheduler = create_scheduler(optimizer, args.warmup_steps, total_steps) if total_steps > 0 else None

    scaler = GradScaler(enabled=args.fp16)

    start_epoch = 0
    if args.resume_from is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled() and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            args.gradient_accumulation_steps,
            args.log_interval,
            is_main_process,
        )

        if is_main_process:
            print(f"[Epoch {epoch}] train_loss = {train_loss:.6f}")

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, is_main_process)
            if is_main_process:
                print(f"[Epoch {epoch}] val_loss = {val_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            save_checkpoint(
                args.output_dir,
                epoch=epoch + 1,
                step=(epoch + 1) * len(train_loader),
                model=module,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                is_main_process=is_main_process,
            )

    cleanup_distributed()


if __name__ == "__main__":
    main()

