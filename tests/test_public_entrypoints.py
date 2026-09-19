from pathlib import Path


def test_public_entrypoints_name_their_scientific_tasks() -> None:
    expected = {
        "validate_dataset.py",
        "fit_defrost_event_models.py",
        "select_defrost_time.py",
        "calculate_rb_reference.py",
        "train_defrost_models.py",
    }
    ambiguous = {
        "main_data.py",
        "main_cost.py",
        "main_labels.py",
        "main_train.py",
        "main_evaluate.py",
    }

    assert all(Path(name).is_file() for name in expected)
    assert all(not Path(name).exists() for name in ambiguous)
