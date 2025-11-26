from pathlib import Path

import torch
from ml_collections import config_flags
from torch import nn
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from torchvision.models import resnet18

from fid import get_cached_loader

_CONFIG_FILE = config_flags.DEFINE_config_file("config", default="fid/configs/train.py")


def main(_):
    cfg = load_cfgs(_CONFIG_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = get_cached_loader(
        shard_glob=cfg.data_dir + "/train/*",
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    val_dataloader = get_cached_loader(
        shard_glob=cfg.data_dir + "/val/*",
        batch_size=cfg.batch_size,
        num_workers=2,
    )

    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if wandb is not None and cfg.wandb_logging.use:
        wandb.init(project=cfg.wandb_logging.project, config=cfg.to_dict())
        if cfg.wandb_logging.log_all:
            wandb.watch(model, log="all")

    global_step = 0
    for epoch in range(cfg.num_epochs):
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            images = batch["img"].unsqueeze(1).to(device)  # add channel dimension
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.4f}")

            if (
                wandb is not None
                and cfg.wandb_logging.use
                and global_step % cfg.wandb_logging.log_interval == 0
            ):
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/accuracy": (outputs.argmax(dim=1) == labels)
                        .float()
                        .mean()
                        .item(),
                    },
                    step=global_step,
                )

            if global_step % cfg.save_interval == 0 and global_step > 0:
                save_path = Path(cfg.checkpoint_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(), save_path / f"resnet18_step{global_step}.pt"
                )

            global_step += 1

        val_loss = 0.0
        val_acc = 0.0
        val_step = 0.0
        model.eval()
        for batch in val_dataloader:
            val_images = batch["img"].unsqueeze(1).to(device)
            val_labels = batch["label"].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels)
                val_acc += (val_outputs.argmax(dim=1) == val_labels).float().mean()
            val_step += 1.0
            if val_step >= cfg.val_steps_per_epoch:
                break

        print(
            f"\nValidation Loss: {val_loss.item()/val_step:.4f}, Accuracy: {val_acc.item()/val_step:.4f}"
        )

        if wandb is not None and cfg.wandb_logging.use and cfg.wandb_logging.use:
            wandb.log(
                {
                    "val/loss": val_loss.item() / val_step,
                    "val/accuracy": val_acc.item() / val_step,
                },
                step=global_step,
            )

        model.train()

        print(f"Epoch {epoch+1}/{cfg.num_epochs} completed.")


def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


if __name__ == "__main__":
    import absl.app as app

    app.run(main)
