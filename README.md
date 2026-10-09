# Transvara

Svara transcription for Carnatic music: a RISE encoder, TCN embedding and GAME
diffusion segmenter fine-tuned jointly to turn a pitch track into note events.

## Setup

```bash
git clone https://github.com/vivekvjyn/transvara.git
cd transvara
pip install -e .
```

Pretrained weights go in `weights/`:

| File | Model |
|---|---|
| `rise_pretrained.pth` | RISE encoder |
| `tcn_carnatic_pretrained.pth` | compIAM TCN Carnatic |
| `game_pretrained_{small,medium,large}.pt` | GAME segmenter |

## Running

```bash
python scripts/train.py
python scripts/eval.py
```

Checkpoints are saved to `weights/runs/<tag>/`, metrics tables and plots to
`results/`. Hyperparameters are in `config.yaml`.

## Licence

MIT
