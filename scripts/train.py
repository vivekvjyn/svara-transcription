from __future__ import annotations

import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
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
    for key, value in {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch size": args.batch_size,
        "chunk": args.chunk,
        "warmup epochs": args.warmup_epochs,
        "lr heads": args.lr_heads,
        "lr encoder": args.lr_encoder,
        "lr segmenter": args.lr_segmenter,
        "weight decay": args.weight_decay,
        "val fraction": args.val_fraction,
        "crops per epoch": args.crops_per_epoch,
        "sample rate": args.sample_rate,
        "audio fps": args.audio_fps,
        "fft size": args.fft_size,
        "hop size": args.hop_size,
        "filter fmin": args.filter_fmin,
        "filter fmax": args.filter_fmax,
        "filter bands": args.filter_bands,
        "log epsilon": args.log_epsilon,
        "seed": args.seed,
        "device": device,
    }.items():
        console.print(f"    {key}: {value}")

    hop = args.hop_size if args.hop_size else args.sample_rate // args.audio_fps

    console.print("\n[bold]Data[/bold]")
    tonics = yaml.safe_load((data_dir / "tonics.yaml").read_text())

    mel_filters = librosa.filters.mel(
        sr=args.sample_rate,
        n_fft=args.fft_size,
        n_mels=args.filter_bands,
        fmin=args.filter_fmin,
        fmax=args.filter_fmax,
    ).T

    performances = []
    for raga in RAGAS:
        for pitch_path in sorted((data_dir / "pitch_tracks" / raga).glob("*.tsv")):
            performer = pitch_path.stem
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
                    console.print(
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

            source = data_dir / "audio" / raga / f"{performer}.mp3"
            audio_cache = cache_dir / "audio" / raga / f"{performer}.pt"
            if audio_cache.exists():
                features = torch.load(audio_cache, map_location="cpu", weights_only=True)
            else:
                console.print(f"extracting audio features from {source}")
                signal, _ = librosa.load(
                    source, sr=args.sample_rate, mono=True
                )
                magnitude = np.abs(
                    librosa.stft(
                        signal,
                        n_fft=args.fft_size,
                        hop_length=hop,
                        win_length=args.fft_size,
                        window=np.hanning(args.fft_size),
                        center=False,
                    )
                ).T
                features = torch.from_numpy(
                    np.log(magnitude @ mel_filters + args.log_epsilon).astype(np.float32)
                )
                audio_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(features, audio_cache)

            num_frames = len(item["times"])
            if features.shape[0] >= num_frames:
                item["audio"] = features[:num_frames]
            else:
                padding = features[-1:].expand(num_frames - features.shape[0], -1)
                item["audio"] = torch.cat([features, padding], dim=0)
            performances.append(item)

    console.print(f"{len(performances)} performances across {len(RAGAS)} ragas")

    train = []
    for raga in RAGAS:
        subset = [p for p in performances if p["raga"] == raga]
        test_performer = TEST_PERFORMERS[raga]
        train.extend(p for p in subset if p["performer"] != test_performer)
    console.print(f"{len(train)} training performances")

    mask_rng = np.random.default_rng(args.seed)
    masks = []
    for performance in train:
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

    console.print("\n[bold]Model[/bold]")
    encoder_checkpoint = weights_dir / "rise" / "rise_pretrained.pth"
    if encoder_checkpoint.exists():
        model.load_pretrained_encoder(encoder_checkpoint, device)
        console.print(f"Loaded RISE encoder weights from {encoder_checkpoint}")
    else:
        console.print(
            f"[yellow]warning:[/yellow] missing pretrained encoder at "
            f"{encoder_checkpoint}; training from scratch"
        )
    audio_checkpoint = weights_dir / "tcn" / "tcn_carnatic_pretrained.pth"
    if not audio_checkpoint.exists():
        raise FileNotFoundError(audio_checkpoint)
    model.load_pretrained_audio_encoder(audio_checkpoint, device)
    console.print(f"Loaded frozen TCN-carnatic audio encoder from {audio_checkpoint}")
    model.apply_lora(
        model_cfg["lora_rank"], model_cfg["lora_alpha"], model_cfg["lora_dropout"]
    )
    console.print(f"{model.num_trainable_parameters:,} trainable parameters")

    run_dir = weights_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Stage 1:[/bold] svara classifier with context")
    classifier_best, classifier_scores = train_classifier(
        model, train, masks, num_classes, args, device, run_dir=run_dir
    )
    console.print(f"Best validation svara F {classifier_best:.4f}")

    console.print("\n[bold]Stage 2:[/bold] GAME segmenter with fused TCN-carnatic audio embeddings")
    segmenter_best, segmenter_scores = train_segmenter(
        model, train, masks, args, device, run_dir=run_dir
    )
    console.print(f"Best validation boundary F {segmenter_best:.4f}")

    for name, scores in (("classifier", classifier_scores), ("segmenter", segmenter_scores)):
        if not scores:
            continue
        figure_path = results_dir / f"{name}_history.png"
        save_history_plot(scores, figure_path)
        console.print(f"figure written to {figure_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train the tsvara transcription model in two stages")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-heads", type=float, default=0.001)
    parser.add_argument("--lr-encoder", type=float, default=0.0001)
    parser.add_argument("--lr-segmenter", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--crops-per-epoch", type=int, default=2000)
    parser.add_argument("--validation-crops", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--audio-fps", type=int, default=100)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--hop-size", type=int, default=None)
    parser.add_argument("--filter-fmin", type=float, default=30.0)
    parser.add_argument("--filter-fmax", type=float, default=17000.0)
    parser.add_argument("--filter-bands", type=int, default=81)
    parser.add_argument("--log-epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def train_classifier(
    model: TranscriptionModel,
    performances: list[dict],
    masks: list[torch.Tensor],
    num_classes: int,
    args,
    device: torch.device,
    *,
    run_dir: Path,
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

    resume = run_dir / "resume_classifier.pt"
    checkpoint = run_dir / "classifier.pth"
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
            console.print(f"[yellow]Ignoring a corrupt resume record at {resume}[/yellow]")
        if stored is not None:
            state.update({key: value for key, value in stored.items() if key in state})
    if state["stopped"] or state["epoch"] >= args.epochs:
        console.print(
            f"Already trained for {state['epoch']} epochs; "
            f"best validation score {state['best_score']:.4f}"
        )
        return float(state["best_score"]), []

    start_epoch = int(state["epoch"])
    if start_epoch and state["model"] is not None and state["optimiser"] is not None:
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        console.print(
            f"Resuming at epoch {start_epoch} with best validation score "
            f"{state['best_score']:.4f}"
        )
    else:
        if start_epoch:
            console.print(
                "The interrupted weights were not saved; restarting this run "
                "from its first epoch"
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
                audio_dim = int(performances[0]["audio"].shape[1])
                inputs = torch.zeros(batch, 2, args.chunk, dtype=torch.float32)
                audio = torch.zeros(batch, args.chunk, audio_dim, dtype=torch.float32)
                targets = torch.zeros(batch, args.chunk, dtype=torch.int64)
                boundaries = torch.zeros(batch, args.chunk, dtype=torch.bool)
                for row, (which, start) in enumerate(batch_specs):
                    stop = start + args.chunk
                    contour = performances[which]["contour"][start:stop]
                    real = contour.shape[0]
                    inputs[row, 0, :real] = contour
                    inputs[row, 1, :real] = performances[which]["voicing"][start:stop]
                    audio[row, :real] = performances[which]["audio"][start:stop]
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

            score = validate_classifier(model, performances, cover, masks, args, device)
            scores.append(float(score))
            console.print(f"classifier epoch {epoch + 1}/{args.epochs} · val {score:.4f}")

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
                    f"Early stopping after {args.patience} epochs without improvement"
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
        console.print(f"Saved the training state of {resume.name} at epoch {epoch}")
        raise

    return best, scores


def train_segmenter(
    model: TranscriptionModel,
    performances: list[dict],
    masks: list[torch.Tensor],
    args,
    device: torch.device,
    *,
    run_dir: Path,
) -> tuple[float, list[float]]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("segmenter.")
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_segmenter,
        weight_decay=args.weight_decay,
    )

    cover = []
    stride = max(int(args.chunk * 0.5), 1)
    for which, performance in enumerate(performances):
        frames = len(performance["targets"])
        for start in range(0, max(frames - args.chunk, 0) + 1, stride):
            cover.append((which, start))
        if frames <= args.chunk:
            cover.append((which, 0))

    resume = run_dir / "resume_segmenter.pt"
    checkpoint = run_dir / "best.pth"
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
            console.print(f"[yellow]Ignoring a corrupt resume record at {resume}[/yellow]")
        if stored is not None:
            state.update({key: value for key, value in stored.items() if key in state})
    if state["stopped"] or state["epoch"] >= args.epochs:
        console.print(
            f"Already trained for {state['epoch']} epochs; "
            f"best validation score {state['best_score']:.4f}"
        )
        return float(state["best_score"]), []

    start_epoch = int(state["epoch"])
    if start_epoch and state["model"] is not None and state["optimiser"] is not None:
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        console.print(
            f"Resuming at epoch {start_epoch} with best validation score "
            f"{state['best_score']:.4f}"
        )
    else:
        if start_epoch:
            console.print(
                "The interrupted weights were not saved; restarting this run "
                "from its first epoch"
            )
        start_epoch = 0
        initial_weights = run_dir / "classifier.pth"
        if initial_weights.exists():
            model.load_state_dict(
                torch.load(initial_weights, map_location=device, weights_only=True)
            )
            console.print(f"Loaded stage 1 weights from {initial_weights}")

    crop_rng = np.random.default_rng(args.seed)
    best = float(state["best_score"])
    stale = int(state["epochs_without_improvement"])
    epoch = start_epoch
    scores: list[float] = []

    try:
        for epoch in range(start_epoch, args.epochs):
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
                audio_dim = int(performances[0]["audio"].shape[1])
                inputs = torch.zeros(batch, 2, args.chunk, dtype=torch.float32)
                audio = torch.zeros(batch, args.chunk, audio_dim, dtype=torch.float32)
                boundaries = torch.zeros(batch, args.chunk, dtype=torch.bool)
                mask = torch.zeros(batch, args.chunk, dtype=torch.bool)
                for row, (which, start) in enumerate(batch_specs):
                    stop = start + args.chunk
                    contour = performances[which]["contour"][start:stop]
                    real = contour.shape[0]
                    inputs[row, 0, :real] = contour
                    inputs[row, 1, :real] = performances[which]["voicing"][start:stop]
                    audio[row, :real] = performances[which]["audio"][start:stop]
                    boundaries[row, :real] = performances[which]["boundaries"][start:stop]
                    mask[row, :real] = True

                outputs = model.segmenter_outputs(
                    inputs.to(device), audio.to(device), boundaries.to(device)
                )
                loss = model.boundary_loss(
                    outputs["boundary_logits"], boundaries.to(device), mask.to(device)
                )

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            score = validate_segmenter(model, performances, cover, masks, args, device)
            scores.append(float(score))
            console.print(f"segmenter epoch {epoch + 1}/{args.epochs} · val {score:.4f}")

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
                    f"Early stopping after {args.patience} epochs without improvement"
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
        console.print(f"Saved the training state of {resume.name} at epoch {epoch}")
        raise

    return best, scores


@torch.no_grad()
def validate_classifier(
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
        audio = torch.zeros(1, args.chunk, performance["audio"].shape[1], dtype=torch.float32)
        targets = torch.zeros(1, args.chunk, dtype=torch.int64)
        boundaries = torch.zeros(1, args.chunk, dtype=torch.bool)
        mask = torch.zeros(1, args.chunk, dtype=torch.bool)
        validation = torch.zeros(1, args.chunk, dtype=torch.bool)
        inputs[0, 0, :real] = contour
        inputs[0, 1, :real] = performance["voicing"][start:stop]
        audio[0, :real] = performance["audio"][start:stop]
        targets[0, :real] = performance["targets"][start:stop]
        boundaries[0, :real] = performance["boundaries"][start:stop]
        mask[0, :real] = True
        validation[0, :real] = masks[which][start:stop]

        frame_mask = validation[0]
        if not frame_mask.any():
            continue
        outputs = model.classifier_outputs(
            inputs.to(device),
            targets.to(device),
            boundaries.to(device),
        )
        labels = outputs["region_logits"].argmax(dim=-1)[outputs["regions"] - 1]
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


@torch.no_grad()
def validate_segmenter(
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
    truth: list[np.ndarray] = []
    predicted: list[np.ndarray] = []

    for index in range(0, len(cover), step):
        which, start = cover[index]
        stop = start + args.chunk
        performance = performances[which]
        contour = performance["contour"][start:stop]
        real = contour.shape[0]
        inputs = torch.zeros(1, 2, args.chunk, dtype=torch.float32)
        audio = torch.zeros(1, args.chunk, performance["audio"].shape[1], dtype=torch.float32)
        boundaries = torch.zeros(1, args.chunk, dtype=torch.bool)
        mask = torch.zeros(1, args.chunk, dtype=torch.bool)
        validation = torch.zeros(1, args.chunk, dtype=torch.bool)
        inputs[0, 0, :real] = contour
        inputs[0, 1, :real] = performance["voicing"][start:stop]
        audio[0, :real] = performance["audio"][start:stop]
        boundaries[0, :real] = performance["boundaries"][start:stop]
        mask[0, :real] = True
        validation[0, :real] = masks[which][start:stop]

        frame_mask = validation[0]
        if not frame_mask.any():
            continue
        outputs = model.transcribe(inputs.to(device), audio.to(device))
        truth.append(boundaries[0][frame_mask].numpy())
        predicted.append(outputs["boundaries"][0][frame_mask].cpu().numpy())

    if was_training:
        model.train()
    if not truth:
        return 0.0
    return float(
        f1_score(
            np.concatenate(truth),
            np.concatenate(predicted),
            zero_division=0.0,
        )
    )


if __name__ == "__main__":
    main()
