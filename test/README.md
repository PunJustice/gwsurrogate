# Performance Benchmarks

This directory includes the performance benchmark used to compare the runtime of
surrogate model evaluations across two git references, such as the current branch
and another branch or pull request.

## What It Measures

The main benchmark script is `benchmark_surrogate_evaluations.py`. It times a
set of surrogate model evaluations, records summary timing data, and can include
profiling information for the slowest or most relevant call paths. The benchmark
can run against the current checkout only, or it can compare the current checkout
with another git reference supplied by the workflow.

The benchmark writes several report files:

- `benchmark_surrogate_evaluations.html`: the human-readable report.
- `benchmark_surrogate_evaluations.json`: structured benchmark results.
- `benchmark_surrogate_evaluations.md`: a Markdown summary.
- `benchmark_surrogate_evaluations.png`: plot output used by the report.

## Running In GitHub Actions

The benchmark is run manually from GitHub Actions:

1. Open the repository on GitHub.
2. Go to `Actions`.
3. Select `GWSurrogate Performance Benchmark`.
4. Click `Run workflow`.

The workflow accepts these inputs:

- `compare_ref`: optional git ref to compare against, such as `testing` or
  `pull/70/head`.
- `compare_label`: optional display name for `compare_ref`, such as `PR-70`.
- `repeat`: number of timing repeats for each benchmark case.
- `number`: number of model evaluations per timing repeat.
- `profile_limit`: number of cProfile rows included for each benchmark case.
- `archive_report`: whether to save this run in the historical GitHub Pages
  archive. The default is `false`.
- `archive_label`: optional label for an archived report, such as
  `NumPy 2 migration baseline`.

## Reports

Each successful run uploads an Actions artifact named `benchmark-results`. The
artifact contains the HTML, JSON, Markdown, and plot outputs for that workflow
run.

The latest successful benchmark report is also published to GitHub Pages:

https://sxs-collaboration.github.io/gwsurrogate/

Archived benchmark reports, when any have been saved, are listed here:

https://sxs-collaboration.github.io/gwsurrogate/runs/

## Archiving Guidelines

Most benchmark runs should not be archived. By default, a run updates the latest
GitHub Pages report and leaves the full output attached to the Actions run as an
artifact.

Set `archive_report` to `true` when the run is worth keeping as a long-term
reference. Good candidates include release baselines, important pull request
comparisons, major dependency changes, major surrogate evaluation changes, and
confirmed performance regressions or fixes.

Leave `archive_report` as `false` for exploratory runs, failed runs, tuning runs,
or runs where the timing environment was known to be noisy. When archiving a
report, use `archive_label` to describe why the result matters.

## Running On HPC Machines

The benchmark can also be run directly on an HPC system when GitHub Actions is
not representative of the target environment. The exact module and conda setup
will vary by system, but the workflow on a machine such as Anvil is:

1. Set up and activate a conda environment with the Python version and system
   libraries needed by `gwsurrogate`.
2. Clone the repository on the HPC filesystem:

   ```sh
   git clone git@github.com:sxs-collaboration/gwsurrogate.git
   cd gwsurrogate
   ```

3. Install the package in that environment:

   ```sh
   python -m pip install .
   ```

4. Start an interactive compute node through the scheduler. Once the job starts,
   record the compute node hostname:

   ```sh
   hostname
   ```

5. From the Anvil login node, connect to that compute node with SSH agent
   forwarding enabled:

   ```sh
   ssh -A HOSTNAME
   ```

   This forwards your SSH login agent so the benchmark script can fetch a
   comparison ref from GitHub, such as a pull request branch.

6. From the repository checkout on the compute node, run the benchmark. For
   example, to compare the current checkout with pull request 70:

   ```sh
   python test/benchmark_surrogate_evaluations.py \
     --fetch-ref pull/70/head \
     --compare-ref FETCH_HEAD \
     --compare-label pr-70
   ```

7. Inspect the generated output files on the HPC filesystem:

   ```sh
   ls test/benchmark_surrogate_evaluations.*
   ```

   The most useful file to open first is
   `test/benchmark_surrogate_evaluations.html`. The JSON and Markdown files are
   useful for downstream analysis or copying results into notes.
