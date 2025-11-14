import torch

class ImageCollatror:
    """Collator for image datasets."""

    def __init__(self) -> None:
        pass

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        images = [sample["image"] for sample in batch]
        images_tensor = torch.stack(images, dim=0)
        return {"images": images_tensor}
    

class QuickDrawImages(torch.utils.data.Dataset):
    """Dataset for QuickDraw images."""

    def __init__(self, root: str, seed: int = 0) -> None:
        self.root = root
        self.seed = seed
        # Load dataset from root directory
        # This is a placeholder implementation
        self.data = self._load_data()

    def _load_data(self):
        # Placeholder for loading data logic
        return []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Placeholder for getting an item
        sample = self.data[idx]
        return {"image": sample}