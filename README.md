# Transvara

Svara transcription for Carnatic music. Stage 1 trains a svara classifier over
a RISE encoder and TCN embedding with pitch context, stage 2 fine-tunes the
GAME diffusion segmenter on fused compIAM TCN Carnatic audio embeddings and the
svara onset/offset annotation.

## Setup

```bash
git clone https://github.com/vivekvjyn/transvara.git
cd transvara
pip install -e .
```

Pretrained weights go in `model_weights/`:

| File | Model |
|---|---|
| `rise/rise_pretrained.pth` | RISE encoder |
| `tcn/tcn_carnatic_pretrained.pth` | compIAM TCN Carnatic |
| `game/game_pretrained.pt` | GAME segmenter |

## Running

```bash
python scripts/train.py
python scripts/eval.py
```

`train.py` runs both stages: the classifier writes `classifier.pth`, then the
segmenter and audio fusion write `best.pth`, both under
`model_weights/runs/`. `eval.py` transcribes the held-out test set and
reports fuzzy alignment, onset/offset and svara F. Metrics tables and plots go
to `results/`. Hyperparameters are in `config.yaml`.

## Licence

MIT
