# Quick, Robot, Draw!

Quick, Robot, Draw! turns Google’s Quick, Draw! sketches into an in-context imitation-learning benchmark. It normalises strokes into token sequences, builds K-shot episodes, and trains three policy families (DiT encoder–decoder, DiT decoder-only, and a bi-LSTM SketchRNN). Prompts for the query sketch can be picked as nearest neighbours in an embedding space built from rasterised sketches.

![Placeholder: model samples](figures/placeholder_samples.gif)

## What you get
- Preprocessing from `.ndjson/.bin` QuickDraw files into centred, scaled sequences (absolute + deltas) with pen/start/sep/reset/stop channels.
- On-the-fly K-shot episodes for in-context imitation learning; optional nearest-neighbour prompt retrieval (cosine similarity on rasterised embeddings with Faiss).
- Ready-to-run configs and scripts for three architectures: DiT encoder–decoder, DiT decoder-only, and bi-LSTM SketchRNN.
- Utilities for rasterising sketches, sampling qualitative outputs, and computing FID-style metrics.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # optional
pip install -r requirements.txt
export PYTHONPATH=.
```

## 1) Download the raw QuickDraw data
Place the official QuickDraw release under `raw/` (or point configs to your location):
```bash
mkdir -p raw
gsutil -m cp 'gs://quickdraw_dataset/full/simplified/*.ndjson' raw/
# or fetch a subset:
gsutil cp 'gs://quickdraw_dataset/full/raw/cat.ndjson' raw/
```

## 2) Build the processed dataset
`scripts/build_dataset.py` preprocesses strokes, stores them in LMDB/WebDataset/HDF5, and writes a manifest. Key config knobs live in `configs/dataset/build_dataset.py` (root paths, backend, K-shot size used for any prebuilt episodes, max token lengths, augmentations).

```bash
PYTHONPATH=. python scripts/build_dataset.py \
  --config configs/dataset/build_dataset.py \
  --config.root data/all-classes/train-val-split/ \
  --config.raw_root raw/ \
  --config.num_workers 8
```

Outputs live under `data/...`:
- `DatasetManifest.json` with counts, split map, and normalization stats.
- `sketches/` (processed absolute + delta coordinates with pen flags).
- `episodes/` if you enable `num_prebuilt_episodes` in the config.

Sanity check a few sketches:
```bash
PYTHONPATH=. python scripts/inspect_dataset.py \
  --config configs/dataset/inspect_dataset.py \
  --config.root data/all-classes/train-val-split/
```
Images are written to `figures/inspect/`.

## 3) Nearest-neighbour prompts (recommended)
EpisodeBuilderSimilar chooses the K prompt sketches closest to the query using cosine similarity on rasterised embeddings.

1. Ensure a ResNet checkpoint for embeddings (default path: `metrics/checkpoints/resnet18_step40000.pt`). Train your own on cached raster images if needed:
   ```bash
   PYTHONPATH=. python metrics/train_resnet18.py --config configs/metrics/train.py
   ```
   (Create cached shards with `scripts/cache_images.py` if you want to retrain.)

2. Compute embeddings for every processed sketch (per family) and store IDs:
   ```bash
   PYTHONPATH=. python metrics/compute_embeddings.py \
     --config configs/metrics/build_faiss_index.py \
     --config.dataset_path data/all-classes/train-val-split/ \
     --config.out_dir metrics/index/ \
     --rasterizer_config configs/metrics/cache.py
   ```

3. Build Faiss indices (inner product over L2-normalised embeddings ⇒ cosine similarity):
   ```bash
   PYTHONPATH=. python metrics/build_faiss_index.py \
     --config configs/metrics/build_faiss_index.py
   ```

Training configs expect `metrics/index/faiss_index/` and `metrics/index/ids_family/` so EpisodeBuilderSimilar can load them. If you prefer random prompts, swap `EpisodeBuilderSimilar` for `EpisodeBuilder` in the loaders.

## 4) Train policies
All commands assume the dataset root is `data/all-classes/train-val-split/` and the Faiss assets are in `metrics/index/`. Override paths as needed.

### DiT decoder-only (in-context diffusion)
Condition on prompts + partial query, denoise the next horizon chunk.
```bash
PYTHONPATH=. python diffusion/train_decoder_only_in_context_imitation_learning.py \
  --config configs/diffusion/decoder_only_in_context_imitation_learning.py \
  --config.data.root data/all-classes/train-val-split/ \
  --config.data.index_dir metrics/index/faiss_index/ \
  --config.data.ids_dir metrics/index/ids_family/ \
  --config.checkpoint.dir diffusion/checkpoints/decoder_only
```

### DiT encoder–decoder (context/query split)
Encodes prompts, decodes the query segment with cross-attention.
```bash
PYTHONPATH=. python diffusion/train_encoder_decoder_in_context_imitation_learning.py \
  --config configs/diffusion/encoder_decoder_in_context_imitation_learning.py \
  --config.data.root data/all-classes/train-val-split/ \
  --config.data.index_dir metrics/index/faiss_index/ \
  --config.data.ids_dir metrics/index/ids_family/ \
  --config.checkpoint.dir diffusion/checkpoints/encoder_decoder
```

### Bi-LSTM SketchRNN (in-context)
Delta-coordinate baseline with mixture density decoder.
```bash
PYTHONPATH=. python lstm/train_in_context_imitation_learning.py \
  --config configs/lstm/in_context_imitation_learning.py \
  --config.data.root data/all-classes/train-val-split/ \
  --config.data.index_dir metrics/index/faiss_index/ \
  --config.data.ids_dir metrics/index/ids_family/ \
  --config.checkpoint.dir lstm/checkpoints
```

For unconditional or single-class training, use the `configs/*/imitation_learning.py` variants (set `K=0` and point `families` in the dataset config to your category list).

## 5) Evaluate and sample
- **Diffusion FID / qualitative grids:** set checkpoint name + epoch in the eval config and run:
  ```bash
  PYTHONPATH=. python diffusion/evaluate_decoder_only_in_context_imitation_learning.py \
    --config configs/diffusion/evaluate_decoder_only_in_context_imitation_learning.py \
    --config.data.root data/all-classes/train-val-split/ \
    --config.data.index_dir metrics/index/faiss_index/ \
    --config.data.ids_dir metrics/index/ids_family/ \
    --config.checkpoint.name policy_epoch_010.pt \
    --config.checkpoint.epoch 10
  ```
  Use the encoder–decoder eval script for that variant. Both rely on `metrics/checkpoints/resnet18_step40000.pt` for features.

- **LSTM sampling:** render PNGs from a trained SketchRNN:
  ```bash
  PYTHONPATH=. python lstm/sample_quickdraw.py \
    --config configs/lstm/sample.py \
    --config.checkpoint lstm/checkpoints/sketchrnn_epoch_010.pt
  ```

`diffusion/sampling.py` also exposes `tokens_to_gif`/`tokens_to_figure` if you want standalone visuals.

## 6) Episode format
Episode tokens are length `T × 7` with:

| Channel | Meaning                              |
|---------|--------------------------------------|
| 0–1     | `dx, dy` deltas (or absolute x, y)   |
| 2       | pen state (1 = draw, 0 = lift)       |
| 3       | start flag                           |
| 4       | separator flag between sketches      |
| 5       | reset flag (between context/query)   |
| 6       | stop flag                            |

Sequence layout for K-shot episodes:
```
[SEP, prompt₁, SEP, ..., promptK, SEP, RESET, SEP, query, STOP]
```
Use `coordinate_mode="absolute"` (diffusion configs) or `"delta"` (LSTM configs).

![Placeholder: episode layout](figures/placeholder_episode.png)
![Placeholder: nearest-neighbour prompts](figures/placeholder_nearest_neighbours.png)

## 7) Repository layout
- `configs/` — ML-Collections configs for dataset, diffusion, LSTM, metrics.
- `dataset/` — preprocessing, storage backends, episode builders, collators, rasterisation.
- `diffusion/` — DiT-based policies, training/eval loops, sampling helpers.
- `lstm/` — SketchRNN model, training, sampling.
- `metrics/` — ResNet feature extractor, FID utilities, embedding + Faiss builders.
- `scripts/` — dataset build/inspect and raster cache helpers.

## License & attribution
- Quick, Draw! data © Google under CC BY 4.0 (see their license/terms).
- Repository code is under the license in `LICENSE`.
