from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tsvara.transcription import TranscriptionModel
from utils import (
    VarnamCorpus,
    Performance,
    apply_plot_style,
    artifact,
    banner,
    build_performances,
    detail,
    load_config,
    parameters_table,
    progress,
    resolve_device,
    rule,
    seed_everything,
    split_performances,
    warn,
)

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


def parse_args(cfg: dict) -> Namespace:
    training = cfg["training"]
    parser = ArgumentParser(description="Fine-tune the tsvara transcription model")
    parser.add_argument("--epochs", type=int, default=training["epochs"])
    parser.add_argument("--patience", type=int, default=training["patience"])
    parser.add_argument("--batch-size", type=int, default=training["batch_size"])
    parser.add_argument("--chunk", type=int, default=training["chunk"])
    parser.add_argument("--warmup-epochs", type=int, default=training["warmup_epochs"])
    parser.add_argument("--lr-heads", type=float, default=training["lr_heads"])
    parser.add_argument("--lr-encoder", type=float, default=training["lr_encoder"])
    parser.add_argument("--weight-decay", type=float, default=training["weight_decay"])
    parser.add_argument("--val-fraction", type=float, default=training["val_fraction"])
    parser.add_argument("--w-svara", type=float, default=training["w_svara"])
    parser.add_argument("--w-boundary", type=float, default=training["w_boundary"])
    parser.add_argument("--crops-per-epoch", type=int, default=training["crops_per_epoch"])
    parser.add_argument("--validation-crops", type=int, default=training["validation_crops"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default="joint")
    return parser.parse_args()


def validation_mask(performance: Performance, fraction: float, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(performance.num_frames, dtype=bool)
    count = int(round(performance.num_frames * fraction))
    mask[rng.choice(performance.num_frames, size=count, replace=False)] = True
    return mask


def validation_frames(
    performances: list[Performance],
    fraction: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [validation_mask(performance, fraction, rng) for performance in performances]


def class_weights(performances: list[Performance], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=float)
    for performance in performances:
        regions = np.cumsum(performance.boundaries)
        index = regions * num_classes + performance.targets
        histogram = np.bincount(index, minlength=(int(regions.max()) + 1) * num_classes)
        counts += histogram.reshape(-1, num_classes).sum(axis=0)
    weights = np.where(counts > 0, counts.sum() / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


class CropBase(Dataset):
    def __init__(self, performances: list[Performance], chunk: int) -> None:
        self.performances = performances
        self.chunk = chunk

    def locate(self, index: int) -> tuple[Performance, int]:
        raise NotImplementedError

    def _crop(self, index: int) -> tuple[torch.Tensor, ...]:
        performance, start = self.locate(index)
        stop = start + self.chunk

        contour = torch.from_numpy(performance.contour[start:stop]).to(torch.float32)
        voicing = torch.from_numpy(performance.voicing[start:stop]).to(torch.float32)
        targets = torch.from_numpy(performance.targets[start:stop]).to(torch.int64)
        boundaries = torch.from_numpy(performance.boundaries[start:stop]).to(torch.bool)
        real = contour.shape[0]

        if real < self.chunk:
            deficit = self.chunk - real
            contour = torch.cat([contour, contour[-1:].expand(deficit, -1)])
            voicing = torch.cat([voicing, voicing[-1:].expand(deficit)])
            targets = torch.cat([targets, targets[-1:].expand(deficit)])
            boundaries = torch.cat([boundaries, torch.zeros(deficit, dtype=torch.bool)])

        mask = torch.zeros(self.chunk, dtype=torch.bool)
        mask[:real] = True
        return torch.stack([contour, voicing]), targets, boundaries, mask

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return self._crop(index)


class RandomCropDataset(CropBase):
    def __init__(
        self,
        performances: list[Performance],
        chunk: int,
        *,
        num_crops: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(performances, chunk)
        self.num_crops = num_crops
        self.rng = np.random.default_rng() if rng is None else rng
        self._lengths = np.array([max(p.num_frames - chunk, 1) for p in performances], dtype=float)

    def __len__(self) -> int:
        return self.num_crops

    def locate(self, index: int) -> tuple[Performance, int]:
        which = int(self.rng.choice(len(self.performances), p=self._lengths / self._lengths.sum()))
        performance = self.performances[which]
        start = int(self.rng.integers(0, max(performance.num_frames - self.chunk, 1)))
        return performance, start


class FixedCropDataset(CropBase):
    def __init__(
        self,
        performances: list[Performance],
        chunk: int,
        crops: list[tuple[int, int]],
    ) -> None:
        super().__init__(performances, chunk)
        self.crops = crops

    def __len__(self) -> int:
        return len(self.crops)

    def locate(self, index: int) -> tuple[Performance, int]:
        which, start = self.crops[index]
        return self.performances[which], start

    @classmethod
    def cover(
        cls,
        performances: list[Performance],
        chunk: int,
        *,
        stride_fraction: float = 0.5,
    ) -> FixedCropDataset:
        stride = max(int(chunk * stride_fraction), 1)
        crops: list[tuple[int, int]] = []
        for which, performance in enumerate(performances):
            for start in range(0, max(performance.num_frames - chunk, 0) + 1, stride):
                crops.append((which, start))
            if performance.num_frames <= chunk:
                crops.append((which, 0))
        return cls(performances, chunk, crops)


@dataclass
class TrainingState:
    epoch: int = 0
    best_score: float = -np.inf
    epochs_without_improvement: int = 0
    stopped: bool = False
    model: dict[str, object] | None = None
    optimiser: dict[str, object] | None = None

    @classmethod
    def load(cls, path: Path) -> TrainingState:
        if not path.exists():
            return cls()
        try:
            stored = torch.load(path, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError):
            warn(f"Ignoring a corrupt resume record at {path}")
            return cls()
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in stored.items() if key in known})

    @property
    def resumable(self) -> bool:
        return self.model is not None and self.optimiser is not None

    def capture(self, model: nn.Module, optimiser: torch.optim.Optimizer) -> None:
        self.model = model.state_dict()
        self.optimiser = optimiser.state_dict()

    def forget_weights(self) -> None:
        self.model = self.optimiser = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".tmp")
        torch.save(vars(self), staged)
        staged.replace(path)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def joint_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    boundaries: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    w_svara: float,
    w_boundary: float,
    boundary_loss: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, torch.Tensor] = {}
    per_region = nn.functional.cross_entropy(
        outputs["region_logits"].reshape(-1, outputs["region_logits"].shape[-1]),
        outputs["region_labels"].reshape(-1),
        weight=weights,
        reduction="none",
    )
    present = outputs["region_mask"].reshape(-1).float()
    losses["svara"] = (per_region * present).sum() / present.sum().clamp_min(1)
    losses["boundary"] = boundary_loss(outputs["boundary_logits"], boundaries, mask)
    total = w_svara * losses["svara"] + w_boundary * losses["boundary"]
    return total, {key: float(value.detach()) for key, value in losses.items()}


def _is_encoder_parameter(model: nn.Module, parameter: nn.Parameter) -> bool:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return False
    return any(parameter is candidate for candidate in encoder.parameters())


def _validation_frames(
    performances: list[Performance],
    masks: list[np.ndarray],
    chunk: int,
) -> list[np.ndarray]:
    set_ = FixedCropDataset.cover(performances, chunk)
    cropped = []
    for which, start in set_.crops:
        stop = start + chunk
        mask = masks[which][start:stop]
        mask = mask if mask.any() else np.zeros(stop - start, dtype=bool)
        cropped.append(mask)
    return cropped


@torch.no_grad()
def validation_score(
    model: TranscriptionModel,
    crops: FixedCropDataset,
    masks: list[np.ndarray],
    device: torch.device,
    validation_crops: int,
) -> float:
    was_training = model.training
    model.eval()

    step = max(len(crops) // validation_crops, 1)
    boundary_true: list[np.ndarray] = []
    boundary_predicted: list[np.ndarray] = []
    label_true: list[np.ndarray] = []
    label_predicted: list[np.ndarray] = []

    for index in range(0, len(crops), step):
        mask = masks[index]
        if not mask.any():
            continue
        inputs, targets, boundaries, _ = crops[index]
        outputs = model.transcribe(inputs.unsqueeze(0).to(device))
        predicted_boundaries = outputs["boundaries"][0].cpu().numpy()
        region_labels = outputs["region_logits"][0].argmax(dim=-1).cpu().numpy()
        regions = outputs["regions"][0].cpu().numpy()
        frame_labels = region_labels[regions - 1]

        boundary_true.append(boundaries.numpy()[mask])
        boundary_predicted.append(predicted_boundaries[mask])
        label_true.append(targets.numpy()[mask])
        label_predicted.append(frame_labels[mask])

    if was_training:
        model.train()
    if not boundary_true:
        return 0.0

    boundary = f1_score(
        np.concatenate(boundary_true), np.concatenate(boundary_predicted), zero_division=0.0
    )
    svara = f1_score(
        np.concatenate(label_true), np.concatenate(label_predicted), average="macro", zero_division=0.0
    )
    return float(0.5 * boundary + 0.5 * svara)


def train_model(
    model: TranscriptionModel,
    train_performances: list[Performance],
    validation_masks: list[np.ndarray],
    num_classes: int,
    args,
    device: torch.device,
    *,
    run_dir: Path,
) -> tuple[float, list[float]]:
    optimiser = torch.optim.Adam(
        [
            {
                "params": [
                    p for p in model.parameters() if p.requires_grad and not _is_encoder_parameter(model, p)
                ],
                "lr": args.lr_heads,
            },
            {
                "params": [p for p in model.parameters() if p.requires_grad and _is_encoder_parameter(model, p)],
                "lr": args.lr_encoder,
            },
        ],
        weight_decay=args.weight_decay,
    )
    weights = class_weights(train_performances, num_classes).to(device)
    checkpoint = run_dir / "best.pth"
    resume = run_dir / "resume.pt"

    train_set = RandomCropDataset(train_performances, args.chunk, num_crops=args.crops_per_epoch)
    validation_set = FixedCropDataset.cover(train_performances, args.chunk)
    validation = _validation_frames(train_performances, validation_masks, args.chunk)

    state = TrainingState.load(resume)
    if state.epoch and not state.resumable:
        warn(f"{run_dir.name}: the interrupted weights were not saved; restarting this run")
        state = TrainingState()
    if state.stopped or state.epoch >= args.epochs:
        detail(f"Already trained for {state.epoch} epochs; best validation score {state.best_score:.4f}")
        return float(state.best_score), []
    if state.epoch:
        model.load_state_dict(state.model)
        optimiser.load_state_dict(state.optimiser)
        detail(f"Resuming at epoch {state.epoch} with best validation score {state.best_score:.4f}")

    scores: list[float] = []
    try:
        with progress() as bar:
            epoch_task = bar.add_task("Epochs", total=args.epochs, completed=state.epoch)
            step_task = bar.add_task("Steps", total=len(train_set), visible=False)

            for epoch in range(state.epoch, args.epochs):
                model.set_encoders_trainable(epoch >= args.warmup_epochs)
                model.train()
                bar.update(step_task, completed=0, visible=True)

                for inputs, targets, boundaries, mask in DataLoader(
                    train_set, batch_size=args.batch_size, shuffle=False
                ):
                    outputs = model.training_outputs(
                        inputs.to(device), targets.to(device), boundaries.to(device)
                    )
                    loss, _ = joint_loss(
                        outputs,
                        targets.to(device),
                        boundaries.to(device),
                        mask.to(device),
                        weights,
                        args.w_svara,
                        args.w_boundary,
                        model.boundary_loss,
                    )
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    bar.advance(step_task)

                score = validation_score(model, validation_set, validation, device, args.validation_crops)
                scores.append(float(score))
                bar.update(
                    epoch_task,
                    advance=1,
                    description=f"Epoch {epoch + 1}/{args.epochs} · val {score:.4f}",
                )
                bar.update(step_task, visible=False)

                if score > state.best_score:
                    state.best_score = score
                    state.epochs_without_improvement = 0
                    save_checkpoint(model, checkpoint)
                else:
                    state.epochs_without_improvement += 1

                state.epoch = epoch + 1
                state.stopped = state.epochs_without_improvement >= args.patience
                state.forget_weights()
                state.save(resume)

                if state.stopped:
                    detail(f"Early stopping after {args.patience} epochs without improvement")
                    break
    except KeyboardInterrupt:
        state.capture(model, optimiser)
        state.save(resume)
        detail(f"Saved the training state of {run_dir.name} at epoch {state.epoch}")
        raise

    return float(state.best_score), scores


def save_history_plot(scores: list[float], path: Path) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    epochs = np.arange(1, len(scores) + 1)
    best = int(np.argmax(scores)) + 1
    figure, axis = plt.subplots(figsize=(5.6, 3.6))
    axis.plot(epochs, scores, color="#1f77b4", label="Validation score")
    axis.plot(
        best,
        scores[best - 1],
        "o",
        color="#d62728",
        markersize=6,
        label=f"Best epoch {best}",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation score")
    axis.set_title("Fine-tuning curve")
    axis.set_xlim(1, max(len(scores), 1))
    axis.grid(axis="y", linewidth=0.4, alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    cfg = load_config()
    args = parse_args(cfg)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    frames_dir = root / ".cache" / "datasets" / "frames"
    weights_dir = root / "weights"
    results_dir = root / "results"
    model_cfg = cfg["model"]
    num_classes = len(SVARAS) + 1

    banner("tsvara train")
    parameters_table(
        {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch size": args.batch_size,
            "chunk": args.chunk,
            "warmup epochs": args.warmup_epochs,
            "lr heads": args.lr_heads,
            "lr encoder": args.lr_encoder,
            "weight decay": args.weight_decay,
            "val fraction": args.val_fraction,
            "w svara": args.w_svara,
            "w boundary": args.w_boundary,
            "seed": args.seed,
            "device": device,
        }
    )

    corpus = VarnamCorpus(
        pitch_track_dir=data_dir / "pitch_tracks",
        annotation_dir=data_dir / "annotations",
        audio_dir=data_dir / "audio",
        tonics_file=data_dir / "tonics.yaml",
    )

    rule("Data")
    performances = build_performances(corpus, RAGAS, SVARAS, frames_dir)
    detail(f"{len(performances)} performances across {len(RAGAS)} ragas")

    train = []
    for raga in RAGAS:
        subset = [p for p in performances if p.raga == raga]
        train_part, _ = split_performances(subset, TEST_PERFORMERS[raga])
        train.extend(train_part)
    detail(f"{len(train)} training performances")

    masks = validation_frames(train, args.val_fraction, args.seed)

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

    rule("Model")
    encoder_checkpoint = weights_dir / "rise_pretrained.pth"
    if encoder_checkpoint.exists():
        model.load_pretrained_encoder(encoder_checkpoint, device)
        detail(f"Loaded RISE encoder weights from {encoder_checkpoint}")
    else:
        warn(f"Missing pretrained encoder at {encoder_checkpoint}; training from scratch")
    model.apply_lora(
        model_cfg["lora_rank"], model_cfg["lora_alpha"], model_cfg["lora_dropout"]
    )
    detail(f"{model.num_trainable_parameters:,} trainable parameters")

    rule("Training")
    run_dir = weights_dir / "runs" / args.run_tag
    best, scores = train_model(model, train, masks, num_classes, args, device, run_dir=run_dir)
    detail(f"Best validation score {best:.4f}")

    if scores:
        results_dir.mkdir(parents=True, exist_ok=True)
        history_path = results_dir / f"{args.run_tag}_history.csv"
        pd.DataFrame(
            {"epoch": np.arange(1, len(scores) + 1), "validation_score": scores}
        ).to_csv(history_path, index=False)
        artifact("Table", history_path)
        figure_path = results_dir / f"{args.run_tag}_history.png"
        save_history_plot(scores, figure_path)
        artifact("Figure", figure_path)


if __name__ == "__main__":
    main()
