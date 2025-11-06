# Quick, Robot, Draw!

**Quick, Robot, Draw!** turns Google’s Quick, Draw! sketches into normalized sequence data ready for *in-context* imitation learning with transformer-based diffusion policies, state-space models, and other sequence learners. The pipeline ingests the official `.ndjson` or `.bin` releases, preprocesses every sketch into absolute and delta trajectories with pen-state channels, assembles configurable K-shot prompt/query episodes, and stores everything in efficient backends with PyTorch-friendly loaders.

---

## The Dataset

- **Consistent geometry:** every sketch is centered, scaled into `[-1, 1]^2`, and available as both absolute points and cumulative deltas.
- **Episode-aware:** episodes follow the structure `[START, prompt₁, SEP, …, RESET, START, query, STOP]` with binary control channels (pen, start, sep, reset, stop) so transformers and diffusion models can consume a single token stream.
- **High-throughput I/O:** supports LMDB, WebDataset shards, or HDF5 for cached sketches/episodes plus deterministic PyTorch `Dataset` + collate utilities.
- **Inspectable + verifiable:** ships with scripts to visualize, profile, and sanity-check the processed cache.

---

## 1. Requirements

- Python 3.9+
- `pip install numpy torch lmdb h5py msgpack PyYAML tqdm matplotlib`
  - Install the appropriate PyTorch wheel for your platform/CUDA setup via [pytorch.org](https://pytorch.org/get-started/locally/).
- `gsutil` for downloading the raw QuickDraw release.

---

## 2. Download the raw Quick, Draw! data

Quick, Robot, Draw! expects the official QuickDraw `.ndjson` or `.bin` files to live under a `raw_root` directory (default `raw/`). Pull whichever categories/splits you need:

```bash
# Install Google Cloud SDK for gsutil if necessary.
mkdir -p raw
gsutil -m cp 'gs://quickdraw_dataset/full/simplified/*.ndjson' raw/
# or selectively:
gsutil cp 'gs://quickdraw_dataset/full/raw/cat.ndjson' raw/
```

You can also download individual files via the [Cloud Storage browser](https://console.cloud.google.com/storage/browser/quickdraw_dataset/), then place them under `raw/`.

---

## 3. Configure the dataset build

`config/data_config.yaml` controls preprocessing and storage:

```yaml
root: "data/"           # where processed caches + manifest live
raw_root: "raw/"        # where the downloaded .ndjson/.bin files live
backend: "lmdb"         # lmdb | webdataset | hdf5
num_prompts: 5          # K-shot size
max_seq_len: 512        # episode token cap
normalize: true         # center & scale each sketch
resample:
  points: null          # optionally force per-stroke point count
augmentations:          # applied online during episode sampling
  rotation: true
  scale: true
  translation: true
storage:
  compression: "zstd"
  shards: 64
max_sketches_per_file: null  # cap sketches pulled from each raw file
families: null               # optionally whitelist specific categories
```

Adjust `raw_root`/`root` to match your filesystem. If you only want a subset, place just those files under `raw_root` or run with `--max-files` to cap the build pass.

---

## 4. Build the processed cache

```bash
PYTHONPATH=. python scripts/build_dataset.py \
    --config config/data_config.yaml \
    --num-workers 4 \
    --max-files 5      # optional while testing
```

This will:

1. Iterate through every `.ndjson`/`.bin` under `raw_root`.
2. Resample strokes (optional), flatten the strokes, and emit pen-up/down markers.
3. Normalize each sketch into `[-1, 1]^2` and compute `(dx, dy)` deltas.
4. Cache both representations plus metadata in the chosen backend (`data/sketches/...`).
5. Write `data/DatasetManifest.json` with counts, normalization stats, and per-family split assignments (train/val/test).
6. Optionally prebuild episodes (`num_prebuilt_episodes`) inside `data/episodes/`.

Use `--force` to rebuild even if a manifest already exists, and `--max-files` to process only the first *N* raw files on a pass.

---

## 5. Episode structure

Each episode contains `K` prompt sketches and one query sketch sampled from the same family:

```
[START, prompt₁, SEP, prompt₂, SEP, …, promptK, SEP, RESET,
 START, query, STOP]
```

![Example Sketches](example_sketches.png)

![Episode Tokens](example_tokens.png)

Tokens are float vectors of width 7:

| Channel | Description                         |
|---------|-------------------------------------|
| 0–1     | `dx, dy` deltas                     |
| 2       | pen state (1 = drawing, 0 = lift)   |
| 3       | start flag                          |
| 4       | separator flag                      |
| 5       | reset flag                          |
| 6       | stop flag                           |

Per-token metadata (family IDs, prompt/query IDs, lengths) accompanies every episode so diffusion/transformer policies can condition on prompts and evaluate queries in-context.

---

## 6. Loading episodes in PyTorch

```python
from dataset.loader import QuickDrawEpisodes, QuickDrawEpisodesAbsolute, quickdraw_collate_fn

dataset = QuickDrawEpisodes(
    root="data/",
    split="train",
    K=5,
    backend="lmdb",
    max_seq_len=512,
    augment=True,
)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=16, collate_fn=quickdraw_collate_fn)

for batch in loader:
    tokens = batch["tokens"]      # (B, T, 7)
    mask = batch["mask"]          # (B, T)
    # feed tokens/mask to transformer, diffusion policy, or SSM
```

Need absolute positions instead of deltas? Use the convenience subclass:

```python
dataset = QuickDrawEpisodesAbsolute(root="data/", split="train", K=5)
```

or pass `coordinate_mode="absolute"` to `QuickDrawEpisodes`.

`QuickDrawEpisodes` assembles episodes lazily from cached sketches, applying deterministic per-worker seeds and optional online augmentations (rotation/scale/translation/jitter). The provided `collate_fn` pads sequences and emits masks for turnkey batching.

#### Diffusion-ready batches

Diffusion transformers that observe the prompts plus the first `S` query tokens and denoise the next `H` tokens can use the `DiffusionCollator` wrapper:

```python
from dataset.loader import QuickDrawEpisodes
from dataset.diffusion import DiffusionCollator
from torch.utils.data import DataLoader

episodes = QuickDrawEpisodes(root="data/", split="train", K=5)
collator = DiffusionCollator(horizon=64)  # randomly samples S per episode
loader = DataLoader(episodes, batch_size=16, collate_fn=collator)

batch = next(iter(loader))
tokens = batch["tokens"]              # padded episode tokens
context_mask = batch["context_mask"]  # prompt + observed query tokens
target_mask = batch["target_mask"]    # denoised segment (length ≤ H)
```

The collator uniformly samples how many query tokens to reveal before denoising, anywhere between `0` and the largest value that still leaves `H` tokens for diffusion. Batch dictionaries now include `observed_query_tokens`, `context_mask`, and `target_mask` while preserving all fields from `quickdraw_collate_fn`.

---

## 7. Utility scripts

| Script | Purpose |
|--------|---------|
| `scripts/visualize_episode.py` | Sample an episode, plot the concatenated trajectory + per-sketch panels, and save PNGs to `figures/`. |
| `scripts/verify_dataset.py` | Validate counts, check for NaNs/shape issues, and sample episodes for sanity. |
| `scripts/profile_loading.py` | Benchmark DataLoader throughput (episodes/sec, tokens/sec). |

Run the scripts with `PYTHONPATH=.` so they can import the package modules.

---

## 8. Storage layout

```
data/
  DatasetManifest.json      # config + stats
  sketches/                 # LMDB/WebDataset/HDF5 backend cache
  episodes/                 # optional prebuilt episodes (same backend)
raw/                        # your downloaded QuickDraw files (input only)
figures/                    # visualizations from visualize_episode.py
```

Switching backends only affects how the `sketches/` and `episodes/` directories are structured—the higher-level APIs stay identical.

---

## 9. Extending beyond Quick, Draw!

The preprocessing + episode builder stack only assumes `"family_id"` and a list of stroke arrays. To plug in datasets like Omniglot or LASA:

1. Implement a raw loader that yields `RawSketch` instances.
2. Reuse `QuickDrawPreprocessor` or subclass it for dataset-specific normalization.
3. Store the processed sketches through `SketchStorage` and use `EpisodeBuilder`/`QuickDrawEpisodes` unchanged.

---

## 10. License & attribution

- The Quick, Draw! dataset is © Google, released under the [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/)—review their [terms](https://github.com/googlecreativelab/quickdraw-dataset#license) before redistribution.
- The tooling in **Quick, Robot, Draw!** is provided under the same license as this repository (see `LICENSE`).
