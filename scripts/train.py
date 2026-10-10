from __future__ import annotations

import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import f1_score

from plots import save_history_plot
from tsvara.transcription import TranscriptionModel

console = Console()

SVARAS: tuple[str, ...] = ("S", "R", "G", "M", "P", "D", "N")

RAGAS: tuple[str, ...] = (
    "abhogi",
    "begada",
    "kalyani",
    "mohanam",
    "sahana",
    "saveri",
    "sri",
)

TEST_PERFORMERS: dict[str, str] = {
    "abhogi": "prasanna",
    "begada": "prasanna",
    "kalyani": "prasanna",
    "mohanam": "prasanna",
    "sahana": "vignesh",
    "saveri": "prasanna",
    "sri": "ramakrishnamurthy",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    args = parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_dir = root / "data"
    cache_dir = root / ".cache" / "datasets"
    weights_dir = root / "model_weights"
    results_dir = root / "results"
    model_cfg = cfg["model"]
    num_classes = len(SVARAS) + 1

    console.print("[bold]tsvara train[/bold]")
    table = Table()
    table.add_column("option")
    table.add_column("value")
    for key, value in {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch size": args.batch_size,
        "chunk": args.chunk,
        "warmup epochs": args.warmup_epochs,
        "lr heads": args.lr_heads,
        "lr encoder": args.lr_encoder,
        "weight decay": args.weight_decay,
        "val fraction": args.val_fraction,
        "crops per epoch": args.crops_per_epoch,
        "seed": args.seed,
        "device": device,
    }.items():
        table.add_row(key, str(value))
    console.print(table)

    console.print("\n[bold]Data[/bold]")
    tonics = yaml.safe_load((data_dir / "tonics.yaml").read_text())

    progress = Progress()
    task = progress.add_task(
        "data",
        total=sum(
            len(list((data_dir / "pitch_tracks" / raga).glob("*.tsv"))) for raga in RAGAS
        ),
    )
    progress.start()
    skipped_notes = []
    performances = []
    for raga in RAGAS:
        for pitch_path in sorted((data_dir / "pitch_tracks" / raga).glob("*.tsv")):
            performer = pitch_path.stem
            progress.update(task, description=f"{raga}/{performer}")
            frames_cache = cache_dir / "frames" / raga / f"{performer}.pt"
            if frames_cache.exists():
                stored = torch.load(frames_cache, map_location="cpu", weights_only=False)
            else:
                tonic = float(tonics[performer])
                track = pd.read_csv(
                    pitch_path, sep="\t", header=None, names=["time", "frequency"]
                )
                times = track["time"].to_numpy(dtype=float)
                frequency = track["frequency"].to_numpy(dtype=float)

                raw = frequency.copy()
                raw[raw == 0] = np.nan
                voicing = (~np.isnan(raw)).astype(float)

                contour = raw.copy()
                is_gap = np.isnan(contour)
                padded = np.concatenate(([False], is_gap, [False]))
                edges = np.flatnonzero(padded[1:] != padded[:-1])
                for start, stop in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
                    if stop - start > 0.02:
                        continue
                    contour[start:stop] = np.nan
                filled = (
                    pd.Series(contour).interpolate(method="linear").ffill().bfill().to_numpy()
                )

                smoothed = np.full_like(filled, np.nan)
                voiced = np.where(~np.isnan(filled))[0]
                if voiced.size:
                    for run in np.split(voiced, np.where(np.diff(voiced) > 1)[0] + 1):
                        run_times, run_cents = times[run], filled[run]
                        if run.size >= 4:
                            low, high = run_cents.min(), run_cents.max()
                            spline = UnivariateSpline(
                                run_times, (run_cents - low) / (high - low), s=0.5
                            )
                            smoothed[run] = spline(run_times) * (high - low) + low
                        elif run.size > 1:
                            smoothed[run] = np.interp(run_times, run_times, run_cents)

                cents = 1200.0 * np.log2(smoothed / tonic)
                cents = (cents + 4200.0) / 8400.0

                annotations = pd.read_csv(
                    data_dir / "annotations" / raga / f"{performer}_{raga}.tsv", sep="\t"
                )
                starts = []
                ends = []
                for begin, finish in zip(
                    annotations["Begin time"], annotations["End time"], strict=True
                ):
                    seconds = 0.0
                    for part in str(begin).split(":"):
                        seconds = seconds * 60 + float(part)
                    starts.append(seconds)
                    seconds = 0.0
                    for part in str(finish).split(":"):
                        seconds = seconds * 60 + float(part)
                    ends.append(seconds)

                targets = np.full(len(times), len(SVARAS), dtype=np.int64)
                onsets = np.zeros(len(times), dtype=bool)
                skipped = 0
                for begin, finish, label in zip(
                    starts, ends, annotations["Annotation"].to_numpy(), strict=True
                ):
                    svara = str(label)[0]
                    if svara not in SVARAS:
                        skipped += 1
                        continue
                    frames = np.where((times >= begin) & (times < finish))[0]
                    if frames.size == 0:
                        continue
                    targets[frames] = SVARAS.index(svara)
                    onsets[frames[0]] = True

                changes = np.zeros(len(times), dtype=bool)
                changes[1:] = targets[1:] != targets[:-1]
                boundaries = changes | onsets
                boundaries[0] = False

                if skipped:
                    skipped_notes.append(
                        f"{raga}/{performer}: skipped {skipped} annotations "
                        "outside the seven svaras"
                    )

                stored = {
                    "tonic": tonic,
                    "times": torch.from_numpy(times),
                    "contour": torch.from_numpy(cents.astype(np.float32)),
                    "voicing": torch.from_numpy(voicing.astype(np.float32)),
                    "targets": torch.from_numpy(targets),
                    "boundaries": torch.from_numpy(boundaries),
                }
                frames_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(stored, frames_cache)

            item = {
                "raga": raga,
                "performer": performer,
                "tonic": float(stored["tonic"]),
                "times": stored["times"],
                "contour": stored["contour"].to(torch.float32),
                "voicing": stored["voicing"].to(torch.float32),
                "targets": stored["targets"].to(torch.int64),
                "boundaries": stored["boundaries"].to(torch.bool),
            }

            performances.append(item)
            progress.advance(task)

    progress.stop()
    for message in skipped_notes:
        console.print(message)
    console.print(f"{len(performances)} performances across {len(RAGAS)} ragas")

    console.print("\n[bold]Model[/bold]")
    encoder_checkpoint = weights_dir / "rise" / "rise_pretrained.pth"
    audio_checkpoint = weights_dir / "tcn" / "tcn_carnatic_pretrained.pth"
    if not audio_checkpoint.exists():
        raise FileNotFoundError(audio_checkpoint)
    if encoder_checkpoint.exists():
        console.print(f"Loaded RISE encoder weights from {encoder_checkpoint}")
    else:
        console.print(
            f"[yellow]warning:[/yellow] missing pretrained encoder at "
            f"{encoder_checkpoint}; training from scratch"
        )
    console.print(f"Loaded frozen TCN-carnatic audio encoder from {audio_checkpoint}")

    results_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Training[/bold]")
    summary = Table(title="Raga finetuning")
    summary.add_column("raga")
    summary.add_column("train performances")
    summary.add_column("best val F")
    for raga in RAGAS:
        training = [
            performance
            for performance in performances
            if performance["raga"] == raga
            and performance["performer"] != TEST_PERFORMERS[raga]
        ]
        mask_rng = np.random.default_rng(args.seed)
        masks = []
        for performance in training:
            frames = len(performance["targets"])
            mask = torch.zeros(frames, dtype=torch.bool)
            count = int(round(frames * args.val_fraction))
            index = torch.from_numpy(mask_rng.choice(frames, size=count, replace=False))
            mask[index] = True
            masks.append(mask)

        model = TranscriptionModel(
            num_classes,
            embed_dim=model_cfg["embed_dim"],
            depth=model_cfg["depth"],
            tcn_filters=model_cfg["tcn_filters"],
            tcn_kernel=model_cfg["tcn_kernel"],
            tcn_dilations=model_cfg["tcn_dilations"],
            dropout=model_cfg["dropout"],
            segmenter_depth=model_cfg["segmenter_depth"],
            boundary_std=model_cfg["boundary_std"],
            boundary_threshold=model_cfg["boundary_threshold"],
            boundary_radius=model_cfg["boundary_radius"],
            d3pm_steps=model_cfg["d3pm_steps"],
        ).to(device)
        if encoder_checkpoint.exists():
            model.load_pretrained_encoder(encoder_checkpoint, device)
        model.load_pretrained_audio_encoder(audio_checkpoint, device)
        model.apply_lora(
            model_cfg["lora_rank"], model_cfg["lora_alpha"], model_cfg["lora_dropout"]
        )
        console.print(
            f"\n[bold]{raga}[/bold] · {len(training)} training performances · "
            f"{model.num_trainable_parameters:,} trainable parameters"
        )
        best, scores = train(
            model,
            raga,
            training,
            masks,
            num_classes,
            args,
            device,
            checkpoint_dir=weights_dir / "tsvara" / raga,
            resume_dir=root / ".cache" / "tsvara" / raga,
        )
        summary.add_row(raga, str(len(training)), f"{best:.4f}")
        if scores:
            figure_path = results_dir / f"classifier_history_{raga}.png"
            save_history_plot(scores, figure_path)
            console.print(f"figure written to {figure_path}")
    console.print(summary)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Finetune the RISE encoder and the svara classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-heads", type=float, default=0.001)
    parser.add_argument("--lr-encoder", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--crops-per-epoch", type=int, default=2000)
    parser.add_argument("--validation-crops", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def train(
    model: TranscriptionModel,
    raga: str,
    performances: list[dict],
    masks: list[torch.Tensor],
    num_classes: int,
    args,
    device: torch.device,
    *,
    checkpoint_dir: Path,
    resume_dir: Path,
) -> tuple[float, list[float]]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("encoder.", "embedding.", "classifier."))
    optimiser = torch.optim.Adam(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("encoder.")
                ],
                "lr": args.lr_heads,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("encoder.")
                ],
                "lr": args.lr_encoder,
            },
        ],
        weight_decay=args.weight_decay,
    )
    counts = np.zeros(num_classes, dtype=float)
    for performance in performances:
        regions = np.cumsum(performance["boundaries"].numpy())
        index = regions * num_classes + performance["targets"].numpy()
        histogram = np.bincount(index, minlength=(int(regions.max()) + 1) * num_classes)
        counts += histogram.reshape(-1, num_classes).sum(axis=0)
    weights = np.where(
        counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0
    )
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    cover = []
    stride = max(int(args.chunk * 0.5), 1)
    for which, performance in enumerate(performances):
        frames = len(performance["targets"])
        for start in range(0, max(frames - args.chunk, 0) + 1, stride):
            cover.append((which, start))
        if frames <= args.chunk:
            cover.append((which, 0))

    resume = resume_dir / "resume.pt"
    checkpoint = checkpoint_dir / "best.pth"
    state: dict[str, object] = {
        "epoch": 0,
        "best_score": -np.inf,
        "epochs_without_improvement": 0,
        "stopped": False,
        "model": None,
        "optimiser": None,
    }
    if resume.exists():
        try:
            stored = torch.load(resume, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError):
            stored = None
            console.print(
                f"[yellow]Ignoring a corrupt resume record for {raga} at {resume}[/yellow]"
            )
        if stored is not None:
            state.update({key: value for key, value in stored.items() if key in state})
    if state["stopped"] or state["epoch"] >= args.epochs:
        console.print(
            f"{raga} already trained for {state['epoch']} epochs; "
            f"best validation score {state['best_score']:.4f}"
        )
        return float(state["best_score"]), []

    start_epoch = int(state["epoch"])
    if start_epoch and state["model"] is not None and state["optimiser"] is not None:
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        console.print(
            f"{raga} resuming at epoch {start_epoch} with best validation score "
            f"{state['best_score']:.4f}"
        )
    else:
        if start_epoch:
            console.print(
                f"{raga}: the interrupted weights were not saved; restarting "
                "this run from its first epoch"
            )
        start_epoch = 0

    crop_rng = np.random.default_rng(args.seed)
    best = float(state["best_score"])
    stale = int(state["epochs_without_improvement"])
    epoch = start_epoch
    scores: list[float] = []

    try:
        for epoch in range(start_epoch, args.epochs):
            model.set_encoders_trainable(epoch >= args.warmup_epochs)
            model.train()

            lengths = np.array(
                [max(len(p["targets"]) - args.chunk, 1) for p in performances], dtype=float
            )
            lengths = lengths / lengths.sum()
            specs = []
            for _ in range(args.crops_per_epoch):
                which = int(crop_rng.choice(len(performances), p=lengths))
                frames = len(performances[which]["targets"])
                start = int(crop_rng.integers(0, max(frames - args.chunk, 1)))
                specs.append((which, start))

            for offset in range(0, len(specs), args.batch_size):
                batch_specs = specs[offset : offset + args.batch_size]
                batch = len(batch_specs)
                inputs = torch.zeros(batch, 2, args.chunk, dtype=torch.float32)
                targets = torch.zeros(batch, args.chunk, dtype=torch.int64)
                boundaries = torch.zeros(batch, args.chunk, dtype=torch.bool)
                for row, (which, start) in enumerate(batch_specs):
                    stop = start + args.chunk
                    contour = performances[which]["contour"][start:stop]
                    real = contour.shape[0]
                    inputs[row, 0, :real] = contour
                    inputs[row, 1, :real] = performances[which]["voicing"][start:stop]
                    targets[row, :real] = performances[which]["targets"][start:stop]
                    boundaries[row, :real] = performances[which]["boundaries"][start:stop]

                outputs = model.classifier_outputs(
                    inputs.to(device),
                    targets.to(device),
                    boundaries.to(device),
                )
                per_region = torch.nn.functional.cross_entropy(
                    outputs["region_logits"].reshape(-1, outputs["region_logits"].shape[-1]),
                    outputs["region_labels"].reshape(-1),
                    weight=weights,
                    reduction="none",
                )
                present = outputs["region_mask"].reshape(-1).float()
                loss = (per_region * present).sum() / present.sum().clamp_min(1)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            score = validate(model, performances, cover, masks, args, device)
            scores.append(float(score))
            console.print(f"{raga} · epoch {epoch + 1}/{args.epochs} · val {score:.4f}")

            if score > best:
                best = score
                stale = 0
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1

            stopped = stale >= args.patience
            state.update(
                {
                    "epoch": epoch + 1,
                    "best_score": best,
                    "epochs_without_improvement": stale,
                    "stopped": stopped,
                    "model": None,
                    "optimiser": None,
                }
            )
            resume.parent.mkdir(parents=True, exist_ok=True)
            staged = resume.with_suffix(".tmp")
            torch.save(state, staged)
            staged.replace(resume)

            if stopped:
                console.print(
                    f"{raga} early stopping after {args.patience} epochs "
                    "without improvement"
                )
                break
    except KeyboardInterrupt:
        state.update(
            {
                "epoch": epoch,
                "best_score": best,
                "epochs_without_improvement": stale,
                "stopped": False,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
            }
        )
        resume.parent.mkdir(parents=True, exist_ok=True)
        staged = resume.with_suffix(".tmp")
        torch.save(state, staged)
        staged.replace(resume)
        console.print(
            f"Saved the training state of {resume.name} for {raga} at epoch {epoch}"
        )
        raise

    return best, scores


@torch.no_grad()
def validate(
    model: TranscriptionModel,
    performances: list[dict],
    cover: list[tuple[int, int]],
    masks: list[torch.Tensor],
    args,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    step = max(len(cover) // args.validation_crops, 1)
    truth: list[torch.Tensor] = []
    predicted: list[torch.Tensor] = []

    for index in range(0, len(cover), step):
        which, start = cover[index]
        stop = start + args.chunk
        performance = performances[which]
        contour = performance["contour"][start:stop]
        real = contour.shape[0]
        inputs = torch.zeros(1, 2, args.chunk, dtype=torch.float32)
        targets = torch.zeros(1, args.chunk, dtype=torch.int64)
        boundaries = torch.zeros(1, args.chunk, dtype=torch.bool)
        validation = torch.zeros(1, args.chunk, dtype=torch.bool)
        inputs[0, 0, :real] = contour
        inputs[0, 1, :real] = performance["voicing"][start:stop]
        targets[0, :real] = performance["targets"][start:stop]
        boundaries[0, :real] = performance["boundaries"][start:stop]
        validation[0, :real] = masks[which][start:stop]

        frame_mask = validation[0]
        if not frame_mask.any():
            continue
        outputs = model.classifier_outputs(
            inputs.to(device),
            targets.to(device),
            boundaries.to(device),
        )
        labels = outputs["region_logits"].argmax(dim=-1).gather(1, outputs["regions"] - 1)
        truth.append(targets[0][frame_mask])
        predicted.append(labels[0][frame_mask].cpu())

    if was_training:
        model.train()
    if not truth:
        return 0.0
    return float(
        f1_score(
            torch.cat(truth).numpy(),
            torch.cat(predicted).numpy(),
            average="macro",
            zero_division=0.0,
        )
    )


if __name__ == "__main__":
    main()
