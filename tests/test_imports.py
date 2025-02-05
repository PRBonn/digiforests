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


def test_dependency_imports():
    try:
        import cuml.cluster
        import MinkowskiEngine as ME
        import quickshift

        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import dependency: {e}")


def test_transform_imports():
    try:
        from digiforests_dataloader.transforms import (
            AddOffsets,
            CenterGlobal,
            RandomRotate,
        )

        assert True
    except ImportError:
        pytest.fail("Failed to import transforms from digiforests_dataloader")


def test_forest_pan_seg_import():
    try:
        from forest_pan_seg import MinkUNetPanoptic

        assert True
    except ImportError:
        pytest.fail("Failed to import MinkUNetPanoptic from forest_pan_seg")


def test_tree_dbh_estimation_import():
    try:
        from tree_dbh_estimation import fit_cylinders

        assert True
    except ImportError:
        pytest.fail("Failed to import fit_cylinders from tree_dbh_estimation")
