from pathlib import Path

def test_data_samples_placeholder():
    data_dir = Path(__file__).parent / "data_samples"
    assert data_dir.exists()
