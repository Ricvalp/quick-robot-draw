from typing import Dict, List

from absl import app
from ml_collections import config_flags
from torch.utils.data import DataLoader

from dataset import QuickDrawSketches, SketchStorage, StorageConfig


def load_config(_CONFIG_FILE):
    cfg = _CONFIG_FILE.value

    return cfg


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="configs/tests/test_dataset.py"
)


def test_sketch_dataset(_):

    cfg = load_config(_CONFIG_FILE)

    root = cfg.data.root

    storage_config = StorageConfig(root=root, backend=cfg.data.backend)
    tmp_sketch_storage = SketchStorage(storage_config, mode="r")
    all_families = tmp_sketch_storage.families()
    family_to_samples: Dict[str, List[str]] = {}

    assigned_families = all_families
    for family in assigned_families:
        samples = tmp_sketch_storage.samples_for_family(family)
        if len(samples) > 0:
            family_to_samples[family] = samples

    tmp_sketch_storage.close()

    family_ids = sorted(family_to_samples.keys())

    def sketches_collate_fn(batch):
        sketches = []
        sketch_ids = []
        for item in batch:
            sketches.append(item["tokens"])
            sketch_ids.append(item["sketch_id"])
        return {"sketch": sketches, "sketch_id": sketch_ids}

    for family in family_ids:
        family_samples = family_to_samples[family]

        dataset = QuickDrawSketches(
            family=family, family_samples=family_samples, storage_config=storage_config
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.loader.batch_size,
            num_workers=cfg.loader.num_workers,
            shuffle=False,
            collate_fn=sketches_collate_fn,
        )

        for batch in dataloader:
            sketches = batch["sketch"]
            assert sketches.size(1) == 2

            break


if __name__ == "__main__":

    app.run(test_sketch_dataset)
