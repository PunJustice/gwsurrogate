"""Prepare the GitHub Pages site for benchmark reports."""

from __future__ import annotations

import datetime
import html
import json
import os
import shutil
from pathlib import Path


SITE = Path("_pages")
PUBLISH = Path("_site")
SOURCE = Path("test")
ARTIFACT_FILES = [
    "benchmark_surrogate_evaluations.json",
    "benchmark_surrogate_evaluations.md",
    "benchmark_surrogate_evaluations.png",
]


def copy_report(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        SOURCE / "benchmark_surrogate_evaluations.html",
        destination / "index.html",
    )
    for filename in ARTIFACT_FILES:
        shutil.copyfile(SOURCE / filename, destination / filename)


def archive_current_report(runs_dir: Path) -> None:
    run_id = os.environ["RUN_ID"]
    archive_dir = runs_dir / run_id
    if archive_dir.exists():
        shutil.rmtree(archive_dir)

    copy_report(archive_dir)

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "run_number": os.environ["GITHUB_RUN_NUMBER"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/"
            f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
        ),
        "archive_label": os.environ["ARCHIVE_LABEL"],
        "compare_ref": os.environ["COMPARE_REF"],
        "compare_label": os.environ["COMPARE_LABEL"],
        "workflow_sha": os.environ["GITHUB_SHA"],
    }
    (archive_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def load_reports(runs_dir: Path) -> list[dict[str, str]]:
    reports = []
    for metadata_path in runs_dir.glob("*/metadata.json"):
        try:
            reports.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    reports.sort(key=lambda item: item.get("run_id", ""), reverse=True)
    return reports


def render_history_index(runs_dir: Path, reports: list[dict[str, str]]) -> None:
    rows = []
    for report in reports:
        run_id = report.get("run_id", "")
        label = report.get("archive_label") or report.get("compare_label") or run_id
        compare_ref = report.get("compare_ref") or "default ref"
        created = report.get("created_at_utc", "")
        run_url = report.get("run_url", "")
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(run_id)}/">{html.escape(label)}</a></td>'
            f"<td>{html.escape(created)}</td>"
            f"<td>{html.escape(compare_ref)}</td>"
            f'<td><a href="{html.escape(run_url)}">Actions run</a></td>'
            "</tr>"
        )

    if rows:
        body = "\n".join(rows)
    else:
        body = '<tr><td colspan="4">No benchmark reports have been archived yet.</td></tr>'

    (runs_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GWSurrogate Benchmark History</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
  </style>
</head>
<body>
  <h1>GWSurrogate Benchmark History</h1>
  <p><a href="../">Latest benchmark report</a></p>
  <table>
    <thead>
      <tr>
        <th>Report</th>
        <th>Created</th>
        <th>Compare ref</th>
        <th>Workflow</th>
      </tr>
    </thead>
    <tbody>
{body}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    copy_report(SITE)

    latest = SITE / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    copy_report(latest)

    runs_dir = SITE / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if os.environ["ARCHIVE_REPORT"] == "true":
        archive_current_report(runs_dir)

    render_history_index(runs_dir, load_reports(runs_dir))

    if PUBLISH.exists():
        shutil.rmtree(PUBLISH)
    shutil.copytree(SITE, PUBLISH, ignore=shutil.ignore_patterns(".git"))


if __name__ == "__main__":
    main()
