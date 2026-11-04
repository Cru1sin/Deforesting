"""Build camera inputs and screen comparable experiment cohorts before model fitting."""

import json
import shutil
import subprocess
import sys

import pandas as pd

from dataset_tools import DatasetLoader
from dataset_tools.local_images import RGB_CAMERA_ORDER
from dataset_tools.training_data import publish_training_data
from image_models.admission import audit_inputs
from image_models.cycle_selection import (
    legal_quality_actions,
    screen_cv_cycles,
    screen_published_cycles,
)
from image_models.frost_progression import match_trajectory_frames
from output_layout import camera_outputs

CAMERAS = ("front", *[c for c in RGB_CAMERA_ORDER if c != "front"])


def image_index(dataset, output, cameras=CAMERAS):
    """Screen all candidates before camera availability; extract only causally used frames."""
    loader = DatasetLoader(dataset)
    full = pd.read_parquet(dataset / "training_data/candidate_decisions.parquet")
    screening = screen_cv_cycles(full, loader.list_cycles(), exclude_rb=False)
    screening.to_csv(output / "terminal_screening.csv", index=False)
    candidates = full.loc[
        full.cycle_name.isin(screening.loc[screening.selected, "cycle_name"])
    ].copy()
    ends = screening.set_index("cycle_name").end_time
    candidates["candidate_defrost_time"] = pd.to_datetime(
        candidates.candidate_defrost_time, format="mixed"
    )
    candidates = candidates.loc[
        legal_quality_actions(candidates)
        & candidates.candidate_defrost_time.lt(candidates.cycle_name.map(ends))
    ]
    metadata = loader.load_image_metadata()
    metadata.groupby("camera_role").agg(
        frames=("file_name", "size"), cycles=("cycle_name", "nunique")
    ).to_csv(output / "image_inventory.csv")
    used = []
    for camera in cameras:
        frames = metadata.loc[metadata.camera_role.eq(camera)].assign(indexed=True)
        matches = match_trajectory_frames(candidates, frames, metric="indexed", max_age_seconds=30)
        used.append(
            matches.loc[matches.file_name.notna(), ["cycle_name", "file_name"]]
            .drop_duplicates()
            .assign(camera_role=camera)
        )
    images = pd.concat(used, ignore_index=True)
    images.to_csv(output / "image_inputs.csv", index=False)
    print("[images] " + str(images.groupby("camera_role").size().to_dict()), flush=True)
    return full


def prepare(args):
    output = args.output
    full = image_index(args.dataset, output, args.cameras)
    command = [
        sys.executable,
        "-m",
        "image_models.dinov2_features",
        "--dataset",
        args.dataset,
        "--image-list",
        output / "image_inputs.csv",
        "--camera-roles",
        *args.cameras,
        "--include-rb",
        "--local-only",
        "--device",
        args.device,
        "--batch-size",
        args.batch_size,
        "--n-jobs",
        args.n_jobs,
    ]
    print("[run] " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)
    loader = DatasetLoader(args.dataset)
    settings = loader.training_settings
    cohorts = []
    for camera in args.cameras:
        paths = camera_outputs(output, camera)
        admission = paths["admission"]
        admission.mkdir(parents=True, exist_ok=True)
        if not (admission / "quality_metrics.csv").exists():
            audit_inputs(
                full,
                loader.list_cycles(),
                args.dataset,
                [args.dataset / "image_DINO_feature"],
                admission,
                n_jobs=args.n_jobs,
                camera_role=camera,
            )
        current = {
            **settings,
            "quality": str((admission / "quality_metrics.csv").resolve()),
            "decisions": str(
                (args.dataset / "training_data/candidate_decisions.parquet").resolve()
            ),
            "dataset": str(args.dataset.resolve()),
            "input": f"{camera} RGB -> frozen DINO -> backward <=30s -> unchanged frequency features; no Harmony/reward mask",
        }
        (admission / "admission_settings.json").write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n"
        )
        # Re-screen the entire candidate catalog with this camera's measured quality.
        from image_models.cycle_selection import screen_admission

        screened = screen_admission(admission / "admission_settings.json")
        old = loader.training_cohort(camera)
        changed = set(old.cycle_name) != set(screened.loc[screened.selected, "cycle_name"])
        if len(old):
            changed |= loader.camera_training_settings(camera) != current
        if changed and len(old):
            archive = paths["archive"]
            if not archive.exists():
                archive.mkdir(parents=True)
                suffix = "" if camera == "front" else f"__{camera}"
                for filename in [
                    "cohort.csv",
                    "reference.parquet",
                    f"settings{suffix}.json",
                    f"cycle_screening{suffix}.csv",
                    f"near_optimal_intervals{suffix}.csv",
                    *[f"{name}{suffix}.parquet" for name in old.cycle_name],
                ]:
                    path = args.dataset / "training_data" / filename
                    if path.exists():
                        shutil.copy2(path, archive / filename)
        cohort, _ = publish_training_data(
            args.dataset,
            admission,
            n_jobs=args.n_jobs,
            append=True,
            camera_role=camera,
            refresh_camera=changed and len(old) > 0,
        )
        cohorts.append(cohort)
        print(
            f"[cohort] {camera}: {len(cohort)} cycles, {cohort.experiment_id.nunique()} experiments",
            flush=True,
        )
    memberships = pd.concat(cohorts, ignore_index=True)
    memberships.to_csv(output / "camera_cohorts.csv", index=False)
    common = set.intersection(*[set(c.cycle_name) for c in cohorts])
    shared = memberships.loc[memberships.cycle_name.isin(common)].drop_duplicates("cycle_name")
    shared[["cycle_name", "experiment_id"]].to_csv(output / "intersection.csv", index=False)
    counts = memberships.groupby("camera_role").agg(
        cycles=("cycle_name", "size"), experiments=("experiment_id", "nunique")
    )
    counts["intersection_cycles"] = len(common)
    counts.to_csv(output / "cohort_counts.csv")


def study_cohorts(args):
    """Screen full candidates first, then separate standard and later experiments."""
    screened = {camera: screen_published_cycles(args.dataset, camera) for camera in CAMERAS}
    front = screened["front"]
    standard = set(
        front.loc[front.selected & front.experiment_id.le(args.standard_through), "cycle_name"]
    )
    memberships = []
    for camera, table in screened.items():
        table.to_csv(args.output / f"cycle_screening_{camera}.csv", index=False)
        selected = table.loc[
            table.selected
            & (table.cycle_name.isin(standard) | table.experiment_id.gt(args.standard_through)),
            ["cycle_name", "experiment_id"],
        ].copy()
        selected["data_role"] = selected.experiment_id.map(
            lambda e: "first" if e <= args.standard_through else "second"
        )
        memberships.append(selected.assign(camera_role=camera))
    cohort = pd.concat(memberships, ignore_index=True)
    for role in ["first", "second"]:
        part = cohort.loc[cohort.data_role.eq(role)]
        common = set.intersection(
            *[set(part.loc[part.camera_role.eq(c), "cycle_name"]) for c in CAMERAS]
        )
        cohort.loc[cohort.data_role.eq(role), "common_test"] = cohort.loc[
            cohort.data_role.eq(role), "cycle_name"
        ].isin(common)
    cohort.to_csv(args.output / "camera_cohorts.csv", index=False)
    counts = (
        cohort.loc[cohort.data_role.eq("first")]
        .groupby("camera_role")
        .agg(
            cycles=("cycle_name", "size"),
            experiments=("experiment_id", "nunique"),
            intersection_cycles=("common_test", "sum"),
        )
    )
    counts["all_standard_eligible_cycles"] = pd.Series(
        {
            camera: int((table.selected & table.experiment_id.le(args.standard_through)).sum())
            for camera, table in screened.items()
        }
    )
    counts["outside_front_reference"] = counts.all_standard_eligible_cycles - counts.cycles
    counts["second_test_cycles"] = (
        cohort.loc[cohort.data_role.eq("second")].groupby("camera_role").size()
    )
    counts.to_csv(args.output / "cohort_counts.csv")
    print(
        "[cohorts] standard first test / independent second test\n" + counts.to_string(), flush=True
    )
    return cohort
