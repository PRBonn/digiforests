import typer
import torch
from pathlib import Path
from digiforests_dataloader import DigiForestsDataModule, DigiForestsDataset


def main(data_dir: Path):
    """
    This script showcases two methods to initialize and use the DigiForests dataset:
    1. Using the standard PyTorch DataLoader
    2. Using the PyTorch Lightning DataModule (recommended)

    Args:
        data_dir (Path): Path to the DigiForests dataset directory.
                         Should contain a 'raw/data_split.json' file.

    Note:
        - The PyTorch Lightning method is recommended for integration with Lightning Trainers.
        - This script is for demonstration purposes and does not perform actual training.
    """

    # Method 1: Standard PyTorch DataLoader
    # Useful for custom training loops or non-Lightning frameworks
    dataset = DigiForestsDataset(root=data_dir, split="train")
    dataloader = torch.utils.data.DataLoader(dataset)
    print(f"Standard DataLoader initialized with {len(dataset)} samples")

    # Method 2: PyTorch Lightning DataModule (Recommended)
    # Ideal for use with Lightning Trainers
    datamodule = DigiForestsDataModule(data_dir=data_dir)

    # Optional: Manually prepare data (normally handled by Lightning Trainer)
    datamodule.prepare_data()
    print("Lightning DataModule prepared successfully")

    # Usage example
    # trainer = pl.Trainer()
    # trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    typer.run(main)
