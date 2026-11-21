"""Train and evaluate the waveform CNN with prepared STEAD and QSIS data."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from seismictorch.training.runtime import select_device, run_epoch
from seismictorch.paths import DATA
from seismictorch.models.waveform import WAVEFORM_MODELS, initialize_waveform_model
from seismictorch.data.archives import CLASS_NAMES, load_waveforms, split_indices

from seismictorch.paths import ROOT


def train(args):
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    fine = args.stage == "finetune"
    if fine:
        weights = torch.load(args.init, map_location="cpu", weights_only=True)
        model = initialize_waveform_model(args.model, weights)
    else:
        model = WAVEFORM_MODELS[args.model]()
    model.to(device)

    source = DATA / ("qsis/prepared/all_merged.pth" if fine else "stead/pth/train.pth")
    dataset, groups, source_count = load_waveforms(source, "all" if fine else "train")
    train_ids, val_ids, *test_ids = split_indices(dataset.tensors[1], groups, args.seed)
    training = Subset(dataset, train_ids.tolist())
    validation = Subset(dataset, val_ids.tolist())
    if fine:
        test = Subset(dataset, test_ids[0].tolist())
        test_source = source
    else:
        test_source = DATA / "stead/pth/test.pth"
        test, _, _ = load_waveforms(test_source, "test")

    loader_args = {"batch_size": args.batch_size, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(training, shuffle=True, **loader_args)
    val_loader = DataLoader(validation, **loader_args)
    test_loader = DataLoader(test, **loader_args)
    criterion = nn.CrossEntropyLoss()
    optimizer_type = torch.optim.AdamW if fine else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    args.output.mkdir(parents=True)
    checkpoint = args.output / "best.pth"
    print(f"{args.stage} on {device}: train={len(training)} val={len(validation)} "
          f"test={len(test)} (filtered {source_count - len(dataset)} from {source.name})")
    best = None
    with (args.output / "history.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            training_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
            validation_metrics = run_epoch(model, val_loader, device, criterion)
            record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                      "train": training_metrics, "validation": validation_metrics}
            scheduler.step()

            if best is None or validation_metrics["accuracy"] > best["validation"]["accuracy"]:
                best = record
                pending = checkpoint.with_suffix(".tmp")
                torch.save({name: value.cpu() for name, value in model.state_dict().items()}, pending)
                pending.replace(checkpoint)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"{epoch:3}/{args.epochs}: train {training_metrics['accuracy']:.2%} "
                  f"val {validation_metrics['accuracy']:.2%} "
                  f"val loss {validation_metrics['loss']:.4f}", flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    test_metrics = run_epoch(model, test_loader, device, criterion)
    summary = {
        "stage": args.stage,
        "model": args.model,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "initial_weights": str(args.init.resolve()) if fine else None,
        "train_source": str(source.relative_to(ROOT)),
        "test_source": str(test_source.relative_to(ROOT)),
        "samples": {"train": len(training), "validation": len(validation), "test": len(test)},
        "classes": CLASS_NAMES,
        "best_epoch": best["epoch"],
        "validation": best["validation"],
        "test": test_metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Best epoch {best['epoch']}: validation {best['validation']['accuracy']:.2%}, "
          f"test {test_metrics['accuracy']:.2%}. Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pretrain", "finetune"))
    parser.add_argument("--model", choices=WAVEFORM_MODELS, default="cnn")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output", required=True, type=Path, help="New directory for this run")
    parser.add_argument("--init", type=Path, help="Pretraining best.pth for QSIS fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    fine = args.stage == "finetune"
    if fine != (args.init is not None):
        parser.error("--init is required for fine-tuning and is only used there")
    if args.epochs is None:
        args.epochs = 100 if fine else 50
    if args.batch_size is None:
        args.batch_size = 32 if fine else 256
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr or args.seed < 0:
        parser.error("epochs, batch size and learning rate must be positive; seed must be non-negative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    train(args)


if __name__ == "__main__":
    main()
