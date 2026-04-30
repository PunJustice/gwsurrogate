#!/usr/bin/env python
"""Benchmark time per surrogate model evaluation.

Author: OpenAI GPT-5 Codex

This script times a small set of representative waveform evaluations, stores
machine-readable JSON, writes a Markdown report with profiling appendices, and
writes a PNG timing table.

Typical usage:

    python test/benchmark_surrogate_evaluations.py

Compare the current checkout against a local branch or git ref:

    python test/benchmark_surrogate_evaluations.py --compare-ref my-speedup-branch --compare-label SPEEDUP

Compare against an open GitHub pull request:

    python test/benchmark_surrogate_evaluations.py --fetch-ref pull/123/head --compare-ref FETCH_HEAD --compare-label PR-123

Note: --compare-ref is the git ref that gets checked out for comparison.
--compare-label is only the display label used in the Markdown/PNG reports.
They can be the same for a local branch, but often differ for refs like
FETCH_HEAD, origin/testing, or pull-request refs.

Add another model by adding its input parameters to MODEL_CONFIGS and adding
its name to ENABLED_MODELS.
"""

from __future__ import annotations

import argparse
import cProfile
import datetime as _datetime
import io
import json
import os
import platform
import pstats
import shutil
import subprocess
import sys
import tempfile
import timeit
from pathlib import Path
from typing import Any


PRECESSING_CONFIG = {
    "params": {
        "q": 4,
        "chiA": [-0.2, 0.4, 0.1],
        "chiB": [-0.5, 0.2, -0.4],
    },
    "dimensionless": {"dt": [0.1, 0.5], "f_low": [0, 0.01]},
    "mks": {
        "dt": [1.0 / 4096.0, 1.0 / 8192.0],
        "f_low": [0, 20],
        "extra_kwargs": {"f_ref": 20, "ellMax": None, "M": 70, "dist_mpc": 100, "units": "mks"},
    },
}

HYBRID_CONFIG = {
    "params": {
        "q": 7,
        "chiA": [0, 0, 0.5],
        "chiB": [0, 0, 0.0], # needed to work for NRHybSur2dq15
    },
    "dimensionless": {"dt": [0.1, 0.5], "f_low": [1e-2, 2e-3]},
    "mks": {
        "dt": [1.0 / 4096.0, 1.0 / 8192.0],
        "f_low": [7, 20],
        "extra_kwargs": {"f_ref": 20, "ellMax": None, "M": 70, "dist_mpc": 100, "units": "mks"},
    },
}

PRECESSING_MODELS = ["NRSur7dq4", "SEOBNRv4PHMSur"]
HYBRID_MODELS = ["NRHybSur3dq8", "NRHybSur2dq15", "NRHybSur3dq8_CCE"]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    **{model: PRECESSING_CONFIG for model in PRECESSING_MODELS},
    **{model: HYBRID_CONFIG for model in HYBRID_MODELS},
}
ENABLED_MODELS = PRECESSING_MODELS + HYBRID_MODELS


def dimensionless_cases(dt_values: list[float], f_low_values: list[float]) -> list[dict[str, Any]]:
    """Build the geometric-unit benchmark cases."""
    cases = []
    for dt in dt_values:
        for f_low in f_low_values:
            cases.append(
                {
                    "id": f"geom_dt_{dt:g}_flow_{f_low:g}",
                    "group": "Geometric Units",
                    "header": f"dt = {dt:g}M<br>f_low = {f_low:g}",
                    "kwargs": {"dt": dt, "f_low": f_low},
                }
            )
    return cases


def mks_cases(dt_values: list[float], f_low_values: list[float], extra_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the MKS benchmark cases."""
    cases = []
    for dt in dt_values:
        for f_low in f_low_values:
            cases.append(
                {
                    "id": f"mks_dt_{dt:.12g}_flow_{f_low:g}",
                    "group": "MKS Units (M_tot = 70 M_sun)",
                    "header": f"dt = 1/{round(1.0 / dt):g}s<br>f_low = {f_low:g} Hz",
                    "kwargs": {
                        "dt": dt,
                        "f_low": f_low,
                        **extra_kwargs,
                    },
                }
            )
    return cases


def model_cases(model: str) -> list[dict[str, Any]]:
    """Build benchmark cases for a model from MODEL_CONFIGS."""
    config = model_config(model)
    mks_config = config["mks"]
    dimensionless_config = config["dimensionless"]
    return (
        mks_cases(mks_config["dt"], mks_config["f_low"], mks_config["extra_kwargs"])
        + dimensionless_cases(dimensionless_config["dt"], dimensionless_config["f_low"])
    )


def benchmark_models(gws: Any) -> list[str]:
    """Return enabled benchmark models after validating configuration."""
    catalog_models = set(gws.catalog._surrogate_world)
    missing_from_catalog = [model for model in ENABLED_MODELS if model not in catalog_models]
    missing_configs = [model for model in ENABLED_MODELS if model not in MODEL_CONFIGS]
    if missing_from_catalog:
        raise ValueError(f"Enabled models missing from catalog: {missing_from_catalog}")
    if missing_configs:
        raise ValueError(f"Enabled models missing from MODEL_CONFIGS: {missing_configs}")
    return list(ENABLED_MODELS)


def run_command(args: list[str], cwd: Path | None = None, check: bool = False) -> str:
    """Run a command and return combined stdout/stderr as stripped text."""
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)
    return completed.stdout.strip()


def gitlink_paths(repo_root: Path) -> list[Path]:
    """Return submodule/gitlink paths tracked by the repository."""
    output = run_command(["git", "ls-files", "-s"], cwd=repo_root)
    paths = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "160000":
            paths.append(Path(parts[3]))
    return paths


def copy_populated_gitlinks(source_root: Path, destination_root: Path) -> None:
    """Copy locally populated gitlink directories into a temporary worktree."""
    for relative_path in gitlink_paths(source_root):
        source = source_root / relative_path
        destination = destination_root / relative_path
        if not source.is_dir():
            continue
        shutil.copytree(source, destination, dirs_exist_ok=True)


def symlink_surrogate_downloads(source_root: Path, destination_root: Path) -> None:
    """Symlink downloaded surrogate data into a temporary worktree."""
    relative_path = Path("gwsurrogate") / "surrogate_downloads"
    source = source_root / relative_path
    destination = destination_root / relative_path
    if not source.exists():
        return
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        try:
            destination.rmdir()
        except OSError:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def build_runtime_artifacts(worktree: Path) -> None:
    """Build C extension modules in place inside a temporary worktree."""
    print(f"Building C extensions in {worktree}", flush=True)
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(worktree),
        check=True,
    )


def git_value(args: list[str], repo_root: Path) -> str:
    """Run a git command in repo_root and return a nonempty value or unknown."""
    value = run_command(["git", *args], cwd=repo_root)
    return value if value else "unknown"


def collect_context(repo_root: Path) -> dict[str, Any]:
    """Collect reproducibility context for a benchmark run."""
    context: dict[str, Any] = {
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_count": os.cpu_count(),
        "conda": {
            "prefix": os.environ.get("CONDA_PREFIX"),
            "default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_exe": os.environ.get("CONDA_PYTHON_EXE"),
        },
        "git": {
            "repo_root": str(repo_root),
            "branch": git_value(["branch", "--show-current"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "describe": git_value(["describe", "--always", "--dirty", "--tags"], repo_root),
            "status_short": git_value(["status", "--short"], repo_root),
        },
    }
    if shutil.which("conda"):
        context["conda"]["list"] = run_command(["conda", "list"])
        context["conda"]["env_export"] = run_command(["conda", "env", "export"])
    if shutil.which("lscpu"):
        context["hardware"] = {"lscpu": run_command(["lscpu"])}
    elif shutil.which("sysctl"):
        context["hardware"] = {"sysctl_machdep_cpu": run_command(["sysctl", "-a"])}
    return context


def model_config(model: str) -> dict[str, Any]:
    """Return the configured intrinsic parameters for a model."""
    try:
        return MODEL_CONFIGS[model]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"No input parameters configured for model {model!r}. Known models: {known}") from exc


def evaluate_case(sur: Any, config: dict[str, Any], case: dict[str, Any]) -> Any:
    """Evaluate one surrogate/input-case combination."""
    params = config["params"]
    return sur(params["q"], params["chiA"], params["chiB"], **case["kwargs"])


def profile_case(sur: Any, config: dict[str, Any], case: dict[str, Any], profile_limit: int) -> str:
    """Run cProfile for one surrogate/input-case combination."""
    profiler = cProfile.Profile()
    profiler.enable()
    evaluate_case(sur, config, case)
    profiler.disable()
    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumtime")
    stats.print_stats(profile_limit)
    return output.getvalue()


def run_single_benchmark(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """Run all configured benchmark cases in the current Python process."""
    import gwsurrogate as gws

    models = benchmark_models(gws)
    results: list[dict[str, Any]] = []

    for model in models:
        config = model_config(model)
        cases = model_cases(model)
        print(f"Loading surrogate {model}", flush=True)
        sur = gws.LoadSurrogate(model)
        for case in cases:
            print(f"Timing {model} / {case['id']}", flush=True)
            evaluate_case(sur, config, case)
            timer = timeit.Timer(lambda: evaluate_case(sur, config, case))
            repeats = timer.repeat(repeat=args.repeat, number=args.number)
            per_eval = [elapsed / args.number for elapsed in repeats]
            profile = profile_case(sur, config, case, args.profile_limit)
            results.append(
                {
                    "model": model,
                    "case_id": case["id"],
                    "case_group": case["group"],
                    "case_header": case["header"],
                    "input": {
                        **config["params"],
                        **case["kwargs"],
                    },
                    "timing_seconds": {
                        "repeat": args.repeat,
                        "number": args.number,
                        "per_eval_repeats": per_eval,
                        "best": min(per_eval),
                        "median": sorted(per_eval)[len(per_eval) // 2],
                    },
                    "profile_cumulative": profile,
                }
            )

    return {
        "schema_version": 1,
        "label": args.run_label,
        "models": models,
        "model_configs": {model: model_config(model) for model in models},
        "cases_by_model": {model: model_cases(model) for model in models},
        "settings": {
            "repeat": args.repeat,
            "number": args.number,
            "profile": True,
            "profile_limit": args.profile_limit,
        },
        "context": collect_context(repo_root),
        "results": results,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data to path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON data from path."""
    return json.loads(path.read_text(encoding="utf-8"))


def format_seconds(value: float | None) -> str:
    """Format a timing value for compact report display."""
    if value is None:
        return ""
    return f"{value:.6g}"


def result_lookup(run: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (model, case_id) pairs to timing result dictionaries."""
    return {
        (result["model"], result["case_id"]): result
        for result in run["results"]
    }


def case_timing(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return timing data for a result, or an empty dictionary if absent."""
    if not result:
        return {}
    return result.get("timing_seconds", {})


def cases_for_model(run: dict[str, Any], model: str) -> list[dict[str, Any]]:
    """Return the benchmark cases associated with model in a run."""
    if "cases_by_model" in run:
        return run["cases_by_model"][model]
    return run.get("cases", [])


def case_dt_label(case: dict[str, Any]) -> str:
    """Return a display label for a benchmark case timestep."""
    kwargs = case["kwargs"]
    if kwargs.get("units") == "mks":
        return f"1/{round(1.0 / kwargs['dt']):g} s"
    return f"{kwargs['dt']:g} M"


def case_f_low_label(case: dict[str, Any]) -> str:
    """Return a display label for a benchmark case low frequency."""
    kwargs = case["kwargs"]
    if kwargs.get("units") == "mks":
        return f"{kwargs['f_low']:g} Hz"
    return f"{kwargs['f_low']:g}"


def format_speedup(baseline: float | None, comparison: float | None) -> str:
    """Format a speedup ratio from two timing values."""
    if baseline is None or comparison in (None, 0):
        return ""
    return f"{baseline / comparison:.3g}x"


def model_parameter_label(model: str, benchmark: dict[str, Any]) -> str:
    """Return a concise label describing the configured model parameters."""
    runs = benchmark.get("runs", [benchmark])
    config = runs[0].get("model_configs", {}).get(model, model_config(model))
    params = config.get("params", config)
    mks_kwargs = config.get("mks", {}).get("extra_kwargs", {})
    ell_max = mks_kwargs.get("ellMax")
    return (
        f"{model}: q={params['q']}, chiA={params['chiA']}, "
        f"chiB={params['chiB']}, ellMax={ell_max}"
    )


def png_table_data(benchmark: dict[str, Any]) -> list[list[str]]:
    """Build PNG table rows with grouped run timing columns."""
    rows = []
    runs = benchmark.get("runs", [benchmark])
    run_lookups = [result_lookup(run) for run in runs]
    models = list(dict.fromkeys(model for run in runs for model in run["models"]))
    _, header_bottom = png_table_headers(benchmark)
    column_count = len(header_bottom)

    for model in models:
        rows.append([model_parameter_label(model, benchmark), *[""] * (column_count - 1)])
        cases = cases_for_model(runs[0], model)
        for case in cases:
            row = [model, case["group"], case_dt_label(case), case_f_low_label(case)]
            run_timings = [
                case_timing(lookup.get((model, case["id"])))
                for lookup in run_lookups
            ]
            for timing in run_timings:
                row.extend([format_seconds(timing.get("best")), format_seconds(timing.get("median"))])
            baseline_best = run_timings[0].get("best") if run_timings else None
            for timing in run_timings[1:]:
                row.append(format_speedup(baseline_best, timing.get("best")))
            rows.append(row)
    return rows


def png_table_headers(benchmark: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Build the two-row grouped header for the PNG timing table."""
    runs = benchmark.get("runs", [benchmark])
    top = ["model", "units", "dt", "f_low"]
    bottom = ["", "", "", ""]
    for run in runs:
        top.extend(["", ""])
        bottom.extend(["best (s)", "median (s)"])
    if len(runs) == 2:
        top.append("speedup")
        bottom.append(f"{runs[0]['label']} / {runs[1]['label']}")
    elif len(runs) > 2:
        baseline = runs[0]["label"]
        for run in runs[1:]:
            top.append(f"speedup vs {baseline}")
            bottom.append(run["label"])
    return top, bottom


def is_model_parameter_row(row: list[str]) -> bool:
    """Return whether a PNG table row is a model-parameter separator row."""
    return bool(row and ": q=" in row[0] and "chiA=" in row[0] and "chiB=" in row[0])


def draw_grouped_run_labels(ax: Any, fig: Any, table: Any, runs: list[dict[str, Any]]) -> None:
    """Draw run labels centered over each best/median column pair."""
    from matplotlib.patches import Rectangle

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for run_index, run in enumerate(runs):
        left_column = 4 + 2 * run_index
        right_column = left_column + 1
        left_bbox = table[(0, left_column)].get_window_extent(renderer)
        right_bbox = table[(0, right_column)].get_window_extent(renderer)
        x0_axes, y0_axes = ax.transAxes.inverted().transform((left_bbox.x0, left_bbox.y0))
        x1_axes, y1_axes = ax.transAxes.inverted().transform((right_bbox.x1, right_bbox.y1))
        ax.add_patch(
            Rectangle(
                (x0_axes, y0_axes),
                x1_axes - x0_axes,
                y1_axes - y0_axes,
                facecolor="#d9ead3",
                edgecolor="black",
                linewidth=1,
                transform=ax.transAxes,
                zorder=5,
            )
        )
        ax.text(
            (x0_axes + x1_axes) / 2.0,
            (y0_axes + y1_axes) / 2.0,
            run["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=6,
        )


def draw_model_parameter_rows(ax: Any, fig: Any, table: Any, rows: list[list[str]]) -> None:
    """Draw full-width model-parameter rows over the table cells."""
    from matplotlib.patches import Rectangle

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    last_column = len(rows[0]) - 1
    for row_index, row in enumerate(rows):
        if not is_model_parameter_row(row):
            continue
        left_bbox = table[(row_index, 0)].get_window_extent(renderer)
        right_bbox = table[(row_index, last_column)].get_window_extent(renderer)
        x0_axes, y0_axes = ax.transAxes.inverted().transform((left_bbox.x0, left_bbox.y0))
        x1_axes, y1_axes = ax.transAxes.inverted().transform((right_bbox.x1, right_bbox.y1))
        ax.add_patch(
            Rectangle(
                (x0_axes, y0_axes),
                x1_axes - x0_axes,
                y1_axes - y0_axes,
                facecolor="#fff2cc",
                edgecolor="black",
                linewidth=1,
                transform=ax.transAxes,
                zorder=5,
            )
        )
        ax.text(
            (x0_axes + x1_axes) / 2.0,
            (y0_axes + y1_axes) / 2.0,
            row[0],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=6,
        )


def write_markdown(path: Path, benchmark: dict[str, Any], png_path: Path | None = None) -> None:
    """Write the human-readable Markdown benchmark report."""
    runs = benchmark.get("runs", [benchmark])

    lines = [
        "# GWSurrogate Evaluation Timing",
        "",
        f"Generated: {benchmark['generated_utc']}",
        "",
        "Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.",
    ]
    if png_path is not None:
        lines.extend(["", f"PNG timing table: `{png_path}`"])
    lines.extend(["", "## Summary", ""])
    for run in runs:
        by_key = result_lookup(run)
        lines.extend([f"### {run['label']}", ""])
        for model in run["models"]:
            cases = cases_for_model(run, model)
            case_groups = list(dict.fromkeys(case["group"] for case in cases))
            lines.extend([f"#### {model}", ""])
            for group in case_groups:
                group_cases = [case for case in cases if case["group"] == group]
                lines.extend([f"{group}:", ""])
                for case in group_cases:
                    timing = case_timing(by_key.get((model, case["id"])))
                    lines.append(
                        "- "
                        f"`dt={case_dt_label(case)}`, `f_low={case_f_low_label(case)}`: "
                        f"best `{format_seconds(timing.get('best'))}` s, "
                        f"median `{format_seconds(timing.get('median'))}` s"
                    )
                lines.append("")

    lines.extend(["## Context", ""])
    for run in runs:
        context = run["context"]
        git = context.get("git", {})
        platform_info = context.get("platform", {})
        conda = context.get("conda", {})
        lines.extend(
            [
                f"### {run['label']}",
                "",
                f"- Git branch: `{git.get('branch', 'unknown')}`",
                f"- Git commit: `{git.get('commit', 'unknown')}`",
                f"- Git describe: `{git.get('describe', 'unknown')}`",
                f"- Python: `{context.get('python', {}).get('version', 'unknown')}`",
                f"- Platform: `{platform_info.get('system', '')} {platform_info.get('release', '')} {platform_info.get('machine', '')}`",
                f"- CPU count: `{context.get('cpu_count', 'unknown')}`",
                f"- Conda env: `{conda.get('default_env') or conda.get('prefix') or 'unknown'}`",
                "",
            ]
        )
        status = git.get("status_short")
        if status and status != "unknown":
            lines.extend(["Git status:", "", "```text", status, "```", ""])
        hardware = context.get("hardware", {})
        for key, value in hardware.items():
            lines.extend([f"{key}:", "", "```text", value, "```", ""])

    lines.extend(["## cProfile Appendix", ""])
    for run in runs:
        lines.extend([f"### {run['label']}", ""])
        for result in run["results"]:
            profile = result.get("profile_cumulative")
            if not profile:
                continue
            title = f"{result['model']} / {result['case_id']}"
            lines.extend([f"#### {title}", "", "```text", profile.rstrip(), "```", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png_table(path: Path, benchmark: dict[str, Any]) -> None:
    """Write a PNG table summarizing benchmark timing results."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gwsurrogate-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = benchmark.get("runs", [benchmark])
    header_top, header_bottom = png_table_headers(benchmark)
    rows = [header_top, header_bottom] + png_table_data(benchmark)
    if len(rows) == 2:
        rows.append([""] * len(header_top))

    fig_width = max(13, 1.15 * len(header_top))
    fig_height = max(2.5, 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, _column), cell in table.get_celld().items():
        if row in (0, 1):
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9ead3")
        elif is_model_parameter_row(rows[row]):
            cell.get_text().set_text("")
            cell.set_facecolor("#fff2cc")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fa")
    draw_grouped_run_labels(ax, fig, table, runs)
    draw_model_parameter_rows(ax, fig, table, rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_subprocess_for_ref(
    script_path: Path,
    repo_root: Path,
    ref: str,
    label: str,
    args: argparse.Namespace,
    temp_root: Path,
) -> dict[str, Any]:
    """Benchmark a git ref in a temporary worktree and return its JSON result."""
    worktree = temp_root / label.replace("/", "_")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), ref],
        cwd=str(repo_root),
        check=True,
    )
    copy_populated_gitlinks(repo_root, worktree)
    symlink_surrogate_downloads(repo_root, worktree)
    build_runtime_artifacts(worktree)
    try:
        json_path = temp_root / f"{label.replace('/', '_')}.json"
        md_path = temp_root / f"{label.replace('/', '_')}.md"
        png_path = temp_root / f"{label.replace('/', '_')}.png"
        command = [
            sys.executable,
            str(script_path),
            "--single-run",
            "--repo-root",
            str(worktree),
            "--run-label",
            label,
            "--json-output",
            str(json_path),
            "--md-output",
            str(md_path),
            "--png-output",
            str(png_path),
            "--repeat",
            str(args.repeat),
            "--number",
            str(args.number),
            "--profile-limit",
            str(args.profile_limit),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, cwd=str(worktree), check=True, env=env)
        return load_json(json_path)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=str(repo_root),
            check=False,
        )


def maybe_fetch_ref(repo_root: Path, fetch_ref: str | None) -> None:
    """Fetch an optional ref from origin before comparing git versions."""
    if not fetch_ref:
        return
    subprocess.run(["git", "fetch", "origin", fetch_ref], cwd=str(repo_root), check=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # These names follow Python timeit's terminology: repeat is the number of
    # independent timing measurements, while number is the number of model
    # evaluations inside each measurement. The reported values are divided by
    # number, so they remain seconds per model evaluation. A modest number=3
    # reduces timer overhead for fast cases without making slow cases too costly.
    parser.add_argument("--repeat", type=int, default=5, help="Number of timing repeats per case.")
    parser.add_argument("--number", type=int, default=3, help="Evaluations per timing repeat.")
    parser.add_argument("--profile-limit", type=int, default=40, help="Number of cProfile rows per case.")
    parser.add_argument("--run-label", help="Label used for the current run. Defaults to the current git branch.")
    parser.add_argument("--json-output", type=Path, default=Path("test/benchmark_surrogate_evaluations.json"))
    parser.add_argument("--md-output", type=Path, default=Path("test/benchmark_surrogate_evaluations.md"))
    parser.add_argument("--png-output", type=Path, default=Path("test/benchmark_surrogate_evaluations.png"))
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compare-ref", action="append", default=[], help="Git ref to benchmark in a temporary worktree.")
    parser.add_argument("--compare-label", action="append", default=[], help="Label for the matching --compare-ref.")
    parser.add_argument(
        "--fetch-ref",
        help="Fetch a ref from origin before comparison, e.g. 'pull/123/head' for a GitHub PR.",
    )
    return parser


def main() -> int:
    """Parse command-line arguments and run the requested benchmark workflow."""
    args = build_parser().parse_args()
    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.number < 1:
        raise SystemExit("--number must be at least 1")
    if not args.run_label:
        branch = git_value(["branch", "--show-current"], repo_root)
        args.run_label = branch if branch != "unknown" else "current"

    if args.single_run:
        run = run_single_benchmark(args, repo_root)
        write_json(args.json_output, run)
        wrapped = {"generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(), **run}
        write_markdown(args.md_output, wrapped, args.png_output)
        write_png_table(args.png_output, wrapped)
        return 0

    maybe_fetch_ref(repo_root, args.fetch_ref)
    current = run_single_benchmark(args, repo_root)
    runs = [current]

    if args.compare_ref:
        with tempfile.TemporaryDirectory(prefix="gwsurrogate-bench-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, ref in enumerate(args.compare_ref):
                label = args.compare_label[index] if index < len(args.compare_label) else ref
                runs.append(run_subprocess_for_ref(script_path, repo_root, ref, label, args, temp_root))

    benchmark = {
        "schema_version": 1,
        "generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "cases_by_model": {model: model_cases(model) for model in ENABLED_MODELS},
        "runs": runs,
    }
    write_json(args.json_output, benchmark)
    write_markdown(args.md_output, benchmark, args.png_output)
    write_png_table(args.png_output, benchmark)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.md_output}")
    print(f"Wrote {args.png_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
