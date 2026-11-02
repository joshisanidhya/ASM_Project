from run import find_dataset_path


def test_lyft_stores_found_in_every_layout(tmp_path):
    flat = tmp_path / "a" / "lyft"
    (flat / "train.zarr").mkdir(parents=True)
    assert find_dataset_path("lyft", tmp_path / "a") == flat

    scenes = tmp_path / "b" / "lyft" / "scenes"
    (scenes / "train.zarr").mkdir(parents=True)
    assert find_dataset_path("lyft", tmp_path / "b") == scenes

    kaggle = tmp_path / "c" / "lyft" / "lyft-motion-prediction-autonomous-vehicles" / "scenes"
    (kaggle / "validate.zarr").mkdir(parents=True)
    assert find_dataset_path("lyft", tmp_path / "c") == kaggle

    (tmp_path / "d" / "lyft").mkdir(parents=True)  # present but empty
    assert find_dataset_path("lyft", tmp_path / "d") is None
