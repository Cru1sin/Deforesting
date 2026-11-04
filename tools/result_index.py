"""Build the optional HTML index of published results."""

import os
from html import escape
from urllib.parse import quote

from image_models import checkpoints


def write_output_index():
    """A result-first directory: topic, named file, and direct model downloads."""
    from output_layout import ARCHIVE, CATEGORIES, REPRODUCTION

    ROOT = checkpoints.ROOT
    output = ROOT.parent.parent if ROOT.parent.name == REPRODUCTION else ROOT.parent
    if not output.exists():
        return

    def link(path, label, base):
        return f'<a href="{quote(os.path.relpath(path, base))}">{escape(label)}</a>'

    def page(title, body):
        return (
            f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{escape(title)}</title>'
            "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 24px;line-height:1.7}"
            "a{color:#165c8a}table{border-collapse:collapse;width:100%}td{padding:8px;border-bottom:1px solid #ddd}"
            "input{padding:12px;width:90%;margin:12px 0}summary{cursor:pointer;font-weight:bold}"
            "h2{margin-top:32px}small{color:#555}</style>"
            f'<h1>{escape(title)}</h1><input id="q" placeholder="输入想找的内容：机位、成本、COP、循环编号、模型…">'
            + body
            + '<script>document.getElementById("q").oninput=function(){const q=this.value.toLowerCase();'
            'document.querySelectorAll("tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));'
            'if(q)document.querySelectorAll("details").forEach(d=>d.open=true);};</script></html>'
        )

    def table(paths, base):
        return (
            "<table>"
            + "".join(f"<tr><td>{link(p, p.name, base)}</td></tr>" for p in paths)
            + "</table>"
        )

    def visible(folder):
        return sorted(
            p
            for p in folder.iterdir()
            if not p.name.startswith((".", "~$")) and p.name != "index.html"
        )

    camera_names = dict(
        front="正面",
        left="左侧",
        left_close="左侧近景",
        top="顶部",
        top_close="顶部近景",
        extreme="极端视角",
    )
    model_rows = []
    for settings in sorted(ROOT.glob("*/training/settings.json")):
        folder = settings.parent.parent
        models = checkpoints.checkpoint_files(folder)
        name = folder.name
        if name.startswith("camera_LSTM25_FiLM_20261025_"):
            name = "六机位模型 · " + camera_names[name.removeprefix("camera_LSTM25_FiLM_20261025_")]
        else:
            name = name.rsplit("_2026", 1)[0].replace("_20261013_", " · ").replace("_20261015", "")
        body = "<p>" + link(output / "index.html", "返回总入口", folder) + "</p>"
        body += "<h2>训练配置、循环与评价</h2>" + table(visible(folder / "training"), folder)
        for role in ("evaluation", "diagnostics"):
            if (folder / role).is_dir():
                body += (
                    "<h2>"
                    + ("评价结果" if role == "evaluation" else "机制诊断")
                    + "</h2>"
                    + table(visible(folder / role), folder)
                )
        body += "<h2>模型权重与各折循环名单</h2><table>"
        for model in models:
            body += "<tr><td>" + link(model, model.stem, folder) + "</td><td>"
            for file, label in [
                (model.parent / "cycles.csv", "循环名单"),
                (model.parent, "该折全部记录"),
            ]:
                if file.exists():
                    body += link(file, label, folder) + " "
            body += "</td></tr>"
        (folder / "index.html").write_text(page(name, body + "</table>"))
        model_rows.append((folder / "index.html", name))

    root_body = "<p>先选想找的结果，再打开图或数据。模型、循环名单可直接下载；过程记录集中在复现材料和历史归档。</p>"
    shortcuts = [
        (
            output.parent
            / "dataset/除霜预测模型/三输入Ridge-两阶段-287-1-统一平滑/candidate_model_parameters.json",
            "最新Ridge成本模型",
        ),
        (output.parent / "dataset/training_data/cohort.csv", "循环名单与评价资格"),
        (output.parent / "dataset/training_data/cycle_screening.csv", "循环筛选结果"),
    ]
    root_body += (
        "<p>" + " · ".join(link(p, label, output) for p, label in shortcuts if p.exists()) + "</p>"
    )
    for category in sorted(CATEGORIES):
        directory = output / category
        if not directory.is_dir():
            continue
        category_body = "<p>" + link(output / "index.html", "返回总入口", directory) + "</p>"
        cards = []
        for topic in visible(directory):
            if not topic.is_dir():
                continue
            # Technical and historical trees stay intact; the daily results get file-level navigation.
            if category not in (REPRODUCTION, ARCHIVE):
                paths = visible(topic)
                body = "<p>" + link(directory / "index.html", "返回本类结果", topic) + "</p>"
                body += table(paths, topic)
                (topic / "index.html").write_text(page(topic.name, body))
                target = topic / "index.html"
            else:
                target = topic
            category_body += "<h2>" + link(target, topic.name, directory) + "</h2>"
            if category not in (REPRODUCTION, ARCHIVE):
                category_body += table(visible(topic), directory)
            cards.append(f"<tr><td>{link(target, topic.name, output)}</td></tr>")
        if category == "02_模型与消融":
            category_body += "<h2>19组保留模型：权重、循环名单和训练配置</h2><table>"
            category_body += (
                "".join(f"<tr><td>{link(p, label, directory)}</td></tr>" for p, label in model_rows)
                + "</table>"
            )
            for model in sorted(ROOT.glob("camera_*/checkpoints/*deployment*/*.pkl")):
                category_body += "<p>" + link(model, "部署权重 · " + model.stem, directory) + "</p>"
        (directory / "index.html").write_text(page(category, category_body))
        if category in (REPRODUCTION, ARCHIVE):
            root_body += f"<details><summary>{link(directory / 'index.html', category, output)}</summary><table>{''.join(cards)}</table></details>"
        else:
            root_body += f"<h2>{link(directory / 'index.html', category, output)}</h2><table>{''.join(cards)}</table>"
    (output / "index.html").write_text(page("结果入口 · 按想找的内容打开", root_body))


if __name__ == "__main__":
    write_output_index()
