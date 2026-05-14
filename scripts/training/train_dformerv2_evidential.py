#!/usr/bin/env python3
"""
Train EvidentialDFormerv2 with F-EDL on NYUv2 / SUN-RGBD.

Workflow mirrors `train_esanet_evidential.py`:
  1) Initialise from a stock pretrained DFormerv2 backbone (RGB-D).
  2) Optional CE warmup phase (--ce-warmup 5..10).
  3) Switch to the F-EDL composite loss (Brier-on-moments + p-regulariser).

The dataloader is expected to emit DFormer-compatible batches:
    'rgb':   FloatTensor [3, H, W]   (ImageNet-normalised)
    'depth': FloatTensor [3, H, W]   (depth replicated to 3 channels and
                                      normalised with mean=[0.48]*3, std=[0.28]*3)
    'label': LongTensor  [H, W]      (class id, 255 = ignore)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidential import build_dformerv2_evidential, FEDLSegLoss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-backbone", type=str, default=None,
                    help="Path to stock DFormerv2 backbone weights "
                         "(loaded by DFormer's own init_weights).")
    ap.add_argument("--data-root",  type=str, required=True)
    ap.add_argument("--dataset",    type=str, default="nyuv2",
                    choices=["nyuv2", "sunrgbd"])
    ap.add_argument("--num-classes", type=int, default=40)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--backbone", type=str, default="DFormerv2_L",
                    choices=["DFormerv2_S", "DFormerv2_B", "DFormerv2_L"])
    ap.add_argument("--decoder",  type=str, default="MLPDecoder",
                    choices=["MLPDecoder", "ham", "UPernet", "deeplabv3+", "nl"])
    ap.add_argument("--decoder-embed-dim", type=int, default=512)
    ap.add_argument("--drop-path-rate",    type=float, default=0.1)
    ap.add_argument("--dformer-path", type=str,
                    default=os.environ.get("DFORMER_PATH", ""))

    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch-size",  type=int,   default=8)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--weight-decay",type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int,   default=4)

    ap.add_argument("--ce-warmup",     type=int,   default=5)
    ap.add_argument("--anneal-epochs", type=int,   default=30)
    ap.add_argument("--lambda-kl-max", type=float, default=0.5)
    ap.add_argument("--lambda-p",      type=float, default=1.0)
    ap.add_argument("--edl-form",      type=str,   default="digamma",
                    choices=["digamma", "log", "mse"])
    ap.add_argument("--no-flexible", action="store_true")
    ap.add_argument("--evidence",   type=str, default="softplus",
                    choices=["softplus", "relu", "exp"])

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--print-every", type=int, default=20)
    return ap.parse_args()


def build_dataloaders(args):
    """
    Replace with your DFormer-compatible NYUv2/SUN-RGBD pipeline.
    Yields dicts with 'rgb' [3,H,W], 'depth' [3,H,W], 'label' [H,W].
    """
    from src.datasets.dformer_rgbx import DFormerRGBXDataset  # adjust to your repo
    train = DFormerRGBXDataset(args.data_root, split="train",
                               num_classes=args.num_classes)
    val   = DFormerRGBXDataset(args.data_root, split="val",
                               num_classes=args.num_classes)
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers, pin_memory=True, drop_last=True),
        DataLoader(val,   batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True),
    )


def build_model(args, device):
    if args.dformer_path and args.dformer_path not in sys.path:
        sys.path.insert(0, args.dformer_path)

    model = build_dformerv2_evidential(
        backbone=args.backbone,
        decoder=args.decoder,
        num_classes=args.num_classes,
        decoder_embed_dim=args.decoder_embed_dim,
        drop_path_rate=args.drop_path_rate,
        pretrained_backbone=args.pretrained_backbone,
        flexible=not args.no_flexible,
        evidence=args.evidence,
    ).to(device)
    return model


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    nll = 0.0; n = 0
    for batch in loader:
        rgb   = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        gt    = batch["label"].to(device, non_blocking=True)
        H, W = gt.shape[-2:]
        out = model(rgb, depth, target_size=(H, W))
        pred = out["mu"].argmax(1)
        for c in range(num_classes):
            i = ((pred == c) & (gt == c)).sum()
            u = ((pred == c) | (gt == c)).sum()
            inter[c] += i; union[c] += u
        gt_oh = torch.nn.functional.one_hot(
            gt.clamp(0, num_classes - 1), num_classes
        ).permute(0, 3, 1, 2).float()
        brier = ((out["mu"] - gt_oh) ** 2).sum(1).mean()
        nll += brier.item(); n += 1
    iou = inter / union.clamp_min(1)
    return iou.mean().item(), iou.cpu().numpy(), nll / max(n, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args, device)
    criterion = FEDLSegLoss(
        num_classes=args.num_classes,
        lambda_p=args.lambda_p,
        lambda_kl_max=args.lambda_kl_max,
        anneal_epochs=args.anneal_epochs,
        ce_warmup_epochs=args.ce_warmup,
        edl_form=args.edl_form,
        flexible=not args.no_flexible,
    ).to(device)

    optim = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs
    )

    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            rgb   = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            H, W = label.shape[-2:]

            out = model(rgb, depth, target_size=(H, W))
            losses = criterion(out, label, epoch=epoch)
            loss = losses["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            if it % args.print_every == 0:
                msg = (f"[ep {epoch:03d} it {it:04d}] "
                       f"loss={loss.item():.4f}")
                if "loss_brier" in losses: msg += f"  brier={losses['loss_brier'].item():.4f}"
                if "loss_p"     in losses: msg += f"  p={losses['loss_p'].item():.4f}"
                if "loss_edl"   in losses: msg += f"  edl={losses['loss_edl'].item():.4f}"
                if "loss_kl"    in losses: msg += f"  kl={losses['loss_kl'].item():.4f}"
                if "lambda_kl"  in losses: msg += f"  λ_kl={losses['lambda_kl'].item():.3f}"
                if "uncertainty" in out:
                    msg += f"  ū={out['uncertainty'].mean().item():.3f}"
                print(msg)

        scheduler.step()

        miou, ious, brier = evaluate(model, val_loader, args.num_classes, device)
        print(f"[ep {epoch:03d}] val mIoU={miou:.4f}  Brier={brier:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "miou": miou,
                "brier": brier,
            }, args.out)
            print(f"   ↳ saved {args.out}")

    print(f"Done. best mIoU = {best_miou:.4f}")


if __name__ == "__main__":
    main()
