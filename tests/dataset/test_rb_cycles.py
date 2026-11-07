from pathlib import Path

from dataset_tools.cycle_metadata import cycle_assets


def test_rb_shares_original_but_routes_processed_and_figures():
    regular = cycle_assets("cycle")
    rb = cycle_assets("cycle", rb=True)
    assert regular["original_csv"] == rb["original_csv"]
    assert all(
        Path(value).parent.name == "cycles_RB" for key, value in rb.items() if key != "original_csv"
    )


def test_cv_excludes_rb_but_loader_retains_both():
    from dataset_tools.load_dataset import DatasetLoader
    from image_models.inputs import reliable_rgb_cohort

    loader = DatasetLoader.__new__(DatasetLoader)
    loader._catalog = {
        "cycles": [
            dict(
                cycle_name=name,
                experiment_id="exp",
                status="valid",
                pre_cop_status={"status": "valid"},
                front_rgb_valid=True,
                assets=cycle_assets(name, rb=rb),
            )
            for name, rb in [("long", False), ("rb", True)]
        ]
    }
    assert set(loader.list_cycles().cycle_name) == {"long", "rb"}
    assert reliable_rgb_cohort(loader).cycle_name.tolist() == ["long"]


def test_complete_peak_filter_reuses_the_model_cohort():
    from dataset_tools.load_dataset import DatasetLoader
    from image_models.inputs import reliable_rgb_cohort

    loader = DatasetLoader.__new__(DatasetLoader)
    loader._catalog = {
        "cycles": [
            dict(
                cycle_name=name,
                experiment_id="exp",
                status="valid",
                pre_cop_status={"status": "valid"},
                front_rgb_valid=True,
                cop_peak_quality={"complete": complete},
                assets=cycle_assets(name),
            )
            for name, complete in [("complete", True), ("truncated", False)]
        ]
    }
    assert reliable_rgb_cohort(loader, require_complete_peak=True).cycle_name.tolist() == [
        "complete"
    ]


def test_new_cycles_use_current_rgb_evidence_without_legacy_flags():
    import pandas as pd

    from dataset_tools.load_dataset import DatasetLoader
    from image_models.inputs import reliable_rgb_cohort

    loader = DatasetLoader.__new__(DatasetLoader)
    loader._catalog = {
        "cycles": [
            dict(
                cycle_name=name,
                experiment_id="exp",
                status="valid",
                pre_cop_status={"status": "valid"},
                front_rgb_valid=flag,
                rgb_frost_status="valid",
                cop_peak_quality={"complete": True},
                assets=cycle_assets(name, rb=True),
            )
            for name, flag in [("new", None), ("rejected", False)]
        ]
    }
    loader._image_metadata = pd.DataFrame(
        dict(cycle_name=["new", "rejected"], camera_role=["front", "front"])
    )
    assert reliable_rgb_cohort(
        loader, include_rb=True, require_complete_peak=True
    ).cycle_name.tolist() == ["new"]
