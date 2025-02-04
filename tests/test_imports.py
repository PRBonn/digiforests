import pytest


def test_package_import():
    try:
        import digiforests_dataloader

        assert True
    except ImportError:
        pytest.fail("Failed to import digiforests_dataloader package")


def test_specific_imports():
    try:
        from digiforests_dataloader import (
            DigiForestsDataset,
            DigiForestsDataModule,
            MinkowskiDigiForestsDataModule,
        )

        assert True
    except ImportError:
        pytest.fail(
            "Failed to import DigiForestsDataset, DigiForestsDataModule, or MinkowskiDigiForestsDataModule"
        )
