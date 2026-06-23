from pathlib import Path


def test_required_project_files_exist():
    required_files = [
        "config.yaml",
        "requirements.txt",
        "Dockerfile",
        "src/data_preprocessing.py",
        "src/train.py",
        "src/evaluate.py",
        "src/predict.py",
        "app/main.py",
        "app/streamlit_app.py",
    ]

    for file_path in required_files:
        assert Path(file_path).exists(), f"Missing required file: {file_path}"