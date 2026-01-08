#!/usr/bin/env python3
"""
Evaluate a decoder-only DiT-based diffusion policy on QuickDraw episodes.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import torch
from ml_collections import ConfigDict, config_flags
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.episode_builder import EpisodeBuilderSimilar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dataset.diffusion import InContextDiffusionCollator
from dataset.loader import QuickDrawEpisodes
from diffusion import DiTDiffusionPolicy
from diffusion.sampling import sample_quickdraw_tokens_decoder_only
from metrics import ResNet18FeatureExtractor, compute_fid
from metrics.cache import RasterizerConfig, SketchToImage


def load_config(_CONFIG_FILE: str) -> ConfigDict:

    cfg = _CONFIG_FILE.value

    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_image_grid(
    images: list,
    name: str | None = None,
    output_dir: str | Path | None = None,
    dpi: int = 150,
) -> matplotlib.figure.Figure:
    """Plot images in the squarest possible grid and optionally save to disk."""
    if not images:
        raise ValueError("images must be a non-empty list.")

    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=dpi)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    fig.tight_layout()

    if name is not None:
        if output_dir is None:
            raise ValueError("output_dir must be provided when name is set.")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / name)

    return fig


def _plot_tokens(
    ax,
    tokens: torch.Tensor,
    title: str,
    coordinate_mode: str,
    color: str = "black",
    invert_axis: bool = True,
) -> None:
    """Render `(N, 3)` tokens on the provided axis."""
    array = tokens.detach().cpu().numpy()
    coords = array[:, :2].cumsum(axis=0) if coordinate_mode == "delta" else array[:, :2]
    pen_state = array[:, 2]
    for token_idx in range(1, coords.shape[0]):
        start = coords[token_idx - 1]
        end = coords[token_idx]
        active = pen_state[token_idx] >= 0.5
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color if active else "tab:red",
            linewidth=1.5,
            linestyle="-" if active else "--",
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    if invert_axis:
        ax.invert_yaxis()
    ax.axis("off")


def _extract_valid_context_tokens(
    context: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Drop left padding using the provided mask (mask includes context + horizon)."""
    context_mask = mask[: context.shape[0]].bool()
    return context[context_mask]


def _split_context_prompts(
    ctx_tokens: torch.Tensor, max_prompts: int
) -> list[torch.Tensor]:
    """Split concatenated context tokens into individual prompt sketches."""
    sketches = []
    current = []
    for token in ctx_tokens:
        if token[6] > 0.5:  # stop token
            break
        if token[5] > 0.5:  # reset separates prompts from the query
            if current:
                sketches.append(torch.stack(current))
                current = []
            break
        if token[4] > 0.5:  # separator between context sketches
            if current:
                sketches.append(torch.stack(current))
                current = []
            continue
        current.append(token[[0, 1, 2]])
    if current and len(sketches) < max_prompts:
        sketches.append(torch.stack(current))
    return sketches[:max_prompts]


def _extract_partial_query(ctx_tokens: torch.Tensor) -> torch.Tensor:
    """Return partial query tokens (after the reset token) if present."""
    partial = []
    reached_reset = False
    for token in ctx_tokens:
        if token[6] > 0.5:  # stop token
            break
        if token[5] > 0.5:
            if reached_reset:
                break
            reached_reset = True
            continue
        if reached_reset:
            partial.append(token[[0, 1, 2]])
    if not partial:
        return ctx_tokens.new_zeros((0, 3))
    return torch.stack(partial)


def _log_qualitative_samples(
    policy: DiTDiffusionPolicy,
    eval_iterator: iter,
    cfg: dict,
    device: torch.device,
) -> None:

    context = next(eval_iterator)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed)

    samples = sample_quickdraw_tokens_decoder_only(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos=context,
        generator=generator,
    )

    batch_size = len(samples)
    for idx in range(batch_size):
        ctx_tokens = context["context"][idx]
        ctx_mask = context["mask"][idx]
        valid_ctx = _extract_valid_context_tokens(ctx_tokens, ctx_mask).detach()
        prompts = _split_context_prompts(valid_ctx, cfg.data.K)
        sample_tokens = samples[idx]

        total_plots = len(prompts) + 1
        cols = min(total_plots, 3)
        rows = (total_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for prompt_idx, prompt_tokens in enumerate(prompts):
            _plot_tokens(
                axes[prompt_idx],
                prompt_tokens,
                f"Context {prompt_idx + 1}",
                cfg.data.coordinate_mode,
            )

        _plot_tokens(
            axes[len(prompts)],
            sample_tokens,
            "Sample",
            cfg.data.coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(f"{cfg.logging.dir}/samples_{idx}.png")
        plt.close(fig)


def _log_qualitative_samples_from_partial_sketches(
    policy: DiTDiffusionPolicy,
    eval_iterator: iter,
    cfg: dict,
    device: torch.device,
) -> None:

    context = next(eval_iterator)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed)

    samples = sample_quickdraw_tokens_decoder_only(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos=context,
        generator=generator,
    )

    batch_size = len(samples)
    for idx in range(batch_size):
        ctx_tokens = context["context"][idx]
        ctx_mask = context["mask"][idx]
        valid_ctx = _extract_valid_context_tokens(ctx_tokens, ctx_mask).detach()
        prompts = _split_context_prompts(valid_ctx, cfg.data.K)
        partial_tokens = _extract_partial_query(valid_ctx)
        sample_tokens = samples[idx]

        total_plots = len(prompts) + 1
        cols = min(total_plots, 3)
        rows = (total_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for prompt_idx, prompt_tokens in enumerate(prompts):
            _plot_tokens(
                axes[prompt_idx],
                prompt_tokens,
                f"Context {prompt_idx + 1}",
                cfg.data.coordinate_mode,
            )

        if partial_tokens.numel() > 0:
            _plot_tokens(
                ax=axes[len(prompts)],
                tokens=partial_tokens,
                title="",
                coordinate_mode=cfg.data.coordinate_mode,
                color="green",
                invert_axis=False,
            )

        _plot_tokens(
            ax=axes[len(prompts)],
            tokens=sample_tokens,
            title="Sample",
            coordinate_mode=cfg.data.coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(f"{cfg.logging.dir}/partial_samples_{idx}.png")
        plt.close(fig)


def _log_many_qualitative_samples(
    policy: DiTDiffusionPolicy,
    eval_iterator: iter,
    cfg: dict,
    device: torch.device,
) -> None:

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed)

    context = next(eval_iterator)

    base = {k: v[:1] for k, v in context.items()}
    context = {
        k: v.repeat(cfg.eval.num_many_samples, *([1] * (v.dim() - 1)))
        for k, v in base.items()
    }

    samples = sample_quickdraw_tokens_decoder_only(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos=context,
        generator=generator,
    )

    ctx_tokens = context["context"][0]
    ctx_mask = context["mask"][0]
    valid_ctx = _extract_valid_context_tokens(ctx_tokens, ctx_mask).detach()
    prompts = _split_context_prompts(valid_ctx, cfg.data.K)

    total_plots = len(prompts) + len(samples)
    cols = min(total_plots, 3)
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for prompt_idx, prompt_tokens in enumerate(prompts):
        _plot_tokens(
            axes[prompt_idx],
            prompt_tokens,
            f"Context {prompt_idx + 1}",
            cfg.data.coordinate_mode,
        )

    for sample_idx, sample_tokens in enumerate(samples):
        ax = axes[len(prompts) + sample_idx]
        _plot_tokens(
            ax=ax,
            tokens=sample_tokens,
            title=f"Sample {sample_idx + 1}",
            coordinate_mode=cfg.data.coordinate_mode,
        )

    for ax in axes[total_plots:]:
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(f"{cfg.logging.dir}/partial_samples_many.png")
    plt.close(fig)


@torch.no_grad()
def _compute_fid(
    policy: DiTDiffusionPolicy,
    eval_iterator: iter,
    cfg: dict,
    device: torch.device,
) -> None:

    rasterizer_config = load_config(_RASTERIZER_CONFIG).rasterizer_config
    rasterizer_config = RasterizerConfig(**rasterizer_config)
    sketch_to_image = SketchToImage(rasterizer_config=rasterizer_config)

    embedding_model = ResNet18FeatureExtractor(
        pretrained_checkpoint_path=cfg.eval.fid.resnet_checkpoint_path
    ).to(device)
    embedding_model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed)

    num_samples = math.ceil(cfg.eval.fid.num_samples / cfg.eval.samples)

    samples = []
    for _ in tqdm(range(num_samples), desc="Generating samples for FID computation"):

        context = next(eval_iterator)
        sketches = sample_quickdraw_tokens_decoder_only(
            policy=policy,
            max_tokens=cfg.data.max_query_len,
            demos=context,
            generator=generator,
        )
        samples.extend(sketches)

    def _prompt_from_batch(batch: dict) -> list[torch.Tensor]:
        prompts = []
        context_tokens = batch["context"]
        context_mask = batch["mask"]
        for idx in range(len(context_tokens)):
            valid_ctx = _extract_valid_context_tokens(
                context_tokens[idx], context_mask[idx]
            ).detach()
            gt_sketches = _split_context_prompts(valid_ctx, cfg.data.K)
            prompts.extend(gt_sketches[:1])
        return prompts

    num_gt_samples = math.ceil(cfg.eval.fid.num_gt_samples / cfg.eval.samples)
    gt_samples = []
    for _ in range(num_gt_samples):
        context = next(eval_iterator)
        gt_samples.extend(_prompt_from_batch(context))

    reference_gt_samples = []
    for _ in range(num_gt_samples):
        context = next(eval_iterator)
        reference_gt_samples.extend(_prompt_from_batch(context))

    embeddings = []
    imgs = []
    for sketch in samples:
        img = sketch_to_image(
            {"tokens": sketch.cpu(), "family_id": None, "sketch_id": None}
        )["img"].to(device=device)
        embedding = embedding_model(img.unsqueeze(0))
        imgs.append(img)
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

    gt_embeddings = []
    gt_imgs = []
    for sketch in gt_samples:
        img = sketch_to_image(
            {"tokens": sketch.cpu(), "family_id": None, "sketch_id": None}
        )["img"].to(device=device)
        embedding = embedding_model(img.unsqueeze(0))
        gt_imgs.append(img)
        gt_embeddings.append(embedding)

    gt_embeddings = torch.cat(gt_embeddings, dim=0).cpu().numpy()

    reference_gt_embeddings = []
    for sketch in reference_gt_samples:
        img = sketch_to_image(
            {"tokens": sketch.cpu(), "family_id": None, "sketch_id": None}
        )["img"].to(device=device)
        embedding = embedding_model(img.unsqueeze(0))
        reference_gt_embeddings.append(embedding)

    reference_gt_embeddings = torch.cat(reference_gt_embeddings, dim=0).cpu().numpy()

    plot_image_grid(
        images=[img.squeeze().cpu().numpy() for img in imgs[:64]],
        name="fid_generated_samples.png",
        output_dir=cfg.logging.dir,
    )
    plot_image_grid(
        images=[img.squeeze().cpu().numpy() for img in gt_imgs[:64]],
        name="fid_gt_samples.png",
        output_dir=cfg.logging.dir,
    )

    reference_fid = compute_fid(
        generated_features=reference_gt_embeddings,
        gt_features=gt_embeddings,
    )

    fid = compute_fid(
        generated_features=embeddings,
        gt_features=gt_embeddings,
    )

    print(f"FID: {fid} \nReference FID (GT vs GT): {reference_fid}")


TASKS = {
    "empty_sketches": _log_qualitative_samples,
    "partial_sketches": _log_qualitative_samples_from_partial_sketches,
    "many_samples": _log_many_qualitative_samples,
    "fid": _compute_fid,
}


def run_selected_tasks(tasks, **kwargs):
    for name in tasks:
        if name not in TASKS:
            raise ValueError(f"Unknown task: {name}")
        TASKS[name](**kwargs)


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config",
    default="configs/diffusion/evaluate_decoder_only_in_context_imitation_learning.py",
)

_RASTERIZER_CONFIG = config_flags.DEFINE_config_file(
    "rasterizer_config", default="configs/metrics/cache.py"
)


def main(_) -> None:
    cfg = load_config(_CONFIG_FILE)
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    Path(cfg.logging.dir).mkdir(parents=True, exist_ok=True)

    # Find the last epoch checkpoint
    if not cfg.checkpoint.epoch:
        checkpoint_dir = Path(cfg.checkpoint.dir) / cfg.checkpoint.name
        checkpoint_files = sorted(checkpoint_dir.glob("policy_epoch_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        checkpoint_path = checkpoint_files[-1]
    else:
        checkpoint_path = (
            Path(cfg.checkpoint.dir)
            / cfg.checkpoint.name
            / f"policy_epoch_{cfg.checkpoint.epoch:03d}.pt"
        )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    policy_cfg = checkpoint["config"]

    policy = DiTDiffusionPolicy(policy_cfg).to(device)

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    eval_dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split="val",
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.run.seed,
        coordinate_mode=cfg.data.coordinate_mode,
        builder_cls=EpisodeBuilderSimilar,
        index_dir=cfg.data.index_dir,
        ids_dir=cfg.data.ids_dir,
    )
    eval_collator = InContextDiffusionCollator(
        horizon=policy_cfg.horizon, seed=cfg.run.seed, eval=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval.samples,
        shuffle=True,
        collate_fn=eval_collator,
    )

    eval_iterator = iter(eval_dataloader)

    run_selected_tasks(
        tasks=cfg.eval.get("tasks", []),
        policy=policy,
        eval_iterator=eval_iterator,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    from absl import app

    app.run(main)
