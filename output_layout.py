"""Output ownership: experiments contain folds; shared data and analyses live separately."""

from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "output"
REPRODUCTION = "90_复现材料"
ARCHIVE = "99_历史归档"
CATEGORIES = {
    "01_核心结果",
    "02_模型与消融",
    "03_数据与物理依据",
    "04_汇报材料",
    REPRODUCTION,
    ARCHIVE,
}


def experiment_root(path):
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.parent.name in {"01_视觉模型训练", "training_checkpoint", "experiments"}:
            return parent
    if path.name == "training":
        return path.parent
    return path


def model_output(path):
    """Validate the explicit experiment root without moving it elsewhere."""
    path = Path(path).resolve()
    roots = (OUTPUT / REPRODUCTION / "01_视觉模型训练", PROJECT / "temp/test")
    if not any(path.parent == root for root in roots):
        raise ValueError(
            "Model output must be a named experiment: "
            "output/90_复现材料/01_视觉模型训练/<name> or temp/test/<name>"
        )
    return path


def training_output(path):
    """Use the chosen experiment's training folder; never relocate an experiment."""
    path = Path(path)
    resolved = path.resolve()
    if resolved.is_relative_to(OUTPUT):
        root = experiment_root(resolved)
        model_output(root)
        return root / "training"
    # Secondary tools can use an explicit external workspace or temporary root.
    if path.name == "training":
        return path
    return path / "training"


def analysis_output(path, *, replay=False):
    """Use the exact result directory, rejecting unclassified project outputs."""
    path = Path(path)
    resolved = path.resolve()
    if replay and not any(
        resolved.is_relative_to(root.resolve())
        for root in (OUTPUT / REPRODUCTION / "analysis", PROJECT / "temp/test")
    ):
        raise ValueError("Replay output must be inside output/90_复现材料/analysis or temp/test")
    if resolved.is_relative_to(OUTPUT):
        relative = resolved.relative_to(OUTPUT)
        if len(relative.parts) < 2 or relative.parts[0] not in CATEGORIES:
            raise ValueError("Choose the actual named result directory inside an output category")
    return path


def camera_outputs(output, camera):
    """Resolve every camera-study destination, including temporary archives."""
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT):
        return dict(
            training=output / "models" / camera / "training",
            admission=output / "admission" / camera,
            archive=output / "archive" / camera,
        )
    names = dict(
        front="正面",
        left="左侧",
        left_close="左侧近景",
        top="顶部",
        top_close="顶部近景",
        extreme="极端视角",
    )
    name = (
        f"六机位_{names[camera]}_LSTM" if output.name == "六机位比较" else f"{output.name}_{camera}"
    )
    return dict(
        training=training_output(OUTPUT / REPRODUCTION / "01_视觉模型训练" / name),
        admission=OUTPUT / REPRODUCTION / "02_循环筛选" / output.name / camera,
        archive=OUTPUT / ARCHIVE / "training_data" / output.name / camera,
    )


def prepare_output(args, *, training=False):
    """Resolve the documented stage directory and print it before any writes."""
    for key in ("run", "compare_run", "selection_run"):
        path = getattr(args, key, None)
        if path is not None and (path / "training/settings.json").is_file():
            setattr(args, key, path / "training")
    if hasattr(args, "output") and args.output is not None:
        if training:
            args.output = training_output(args.output)
        else:
            args.output = analysis_output(args.output)
    if getattr(args, "output", None) is not None:
        print(f"Output: {args.output.resolve()}", flush=True)
