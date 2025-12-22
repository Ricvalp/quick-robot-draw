import torch
from absl import app
from ml_collections import config_flags
from torch.utils.data import DataLoader

from dataset import (
    ContextQueryInContextDiffusionCollator,
    InContextSketchRNNCollator,
    QuickDrawEpisodes,
)


def load_config(_CONFIG_FILE):
    cfg = _CONFIG_FILE.value

    return cfg


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="configs/tests/test_dataset.py"
)


def test_collators(_):

    cfg = load_config(_CONFIG_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion_dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split=cfg.data.split,
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.run.seed,
        coordinate_mode="absolute",
    )

    lstm_dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split=cfg.data.split,
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.run.seed,
        coordinate_mode="delta",
    )

    diffusion_collator = ContextQueryInContextDiffusionCollator(
        horizon=cfg.diffusion_model.horizon, seed=cfg.run.seed
    )

    lstm_collator = InContextSketchRNNCollator(
        max_context_len=cfg.data.max_context_len,
        max_query_len=cfg.data.max_query_len,
        teacher_forcing_with_eos=False,
    )

    diffusion_dataloader = DataLoader(
        diffusion_dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        # pin_memory=True,
        # drop_last=True,
        collate_fn=diffusion_collator,
    )
    lstm_dataloader = DataLoader(
        lstm_dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        # pin_memory=True,
        # drop_last=True,
        collate_fn=lstm_collator,
    )

    for diffusion_batch, lstm_batch in zip(diffusion_dataloader, lstm_dataloader):
        diffusion_batch = {k: v.to(device) for k, v in diffusion_batch.items()}
        lstm_batch = {k: v.to(device) for k, v in lstm_batch.items()}
        assert diffusion_batch["context_tokens"].shape[0] == (
            cfg.loader.batch_size,
            ...,
        )
        assert diffusion_batch["query_tokens"].shape[0] == (cfg.loader.batch_size, ...)

        # assert something else...

        break


if __name__ == "__main__":

    app.run(test_collators)
