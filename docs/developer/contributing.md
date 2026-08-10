---
hide:
  - navigation
---

# Contributing

## Environment setup

Requires:

- **Rust** ≥ 1.85 — install via [rustup](https://rustup.rs/) and keep updated with `rustup update`
- **Python** ≥ 3.10
- **CMake** — needed to build the Ceres solver (skip with `--no-default-features` if you don't need fitting features)
- **GSL** (optional) — enables the `"lmsder"` fitting backend; install via your system package manager (`libgsl-dev` on Debian/Ubuntu, `gsl` on Homebrew)

```bash
git clone https://github.com/light-curve/light-curve-python.git
cd light-curve-python/light-curve

python3 -m venv venv && source venv/bin/activate
pip install maturin
maturin develop --group dev   # builds Rust extension + installs dev deps
pre-commit install            # install pre-commit hooks
```

After this initial setup: activate the venv with `source venv/bin/activate`, then rebuild
Rust code with `maturin develop` after any Rust change. Python-only changes need no rebuild.

For a faster iteration cycle (no system deps):

```bash
maturin develop --no-default-features --features=abi3,mimalloc
```

## Running tests

```bash
pytest                                               # all tests (benchmarks disabled by default)
pytest --benchmark-enable                            # with benchmarks
pytest tests/light_curve_ext/test_feature.py        # single test file
pytest -k "Amplitude"                               # filter by name
```

Tests also cover the README examples via `markdown-pytest`.

## Linting and formatting

All linting runs automatically via pre-commit hooks on each commit.

```bash
# Python
ruff check .            # lint
ruff format .           # format

# Rust
cargo fmt --manifest-path=Cargo.toml
cargo clippy --all-targets -- -D warnings   # warnings are errors

# Everything at once
pre-commit run --all-files
```

## Cargo features

| Feature | Default | Effect |
|---------|---------|--------|
| `abi3` | ✓ | Stable CPython ABI3 — disable for PyPy or free-threaded builds (requires Python 3.14t+) |
| `ceres-source` | ✓ | Build Ceres solver from source (requires C++ compiler + CMake) |
| `ceres-system` | | Link against system-installed Ceres instead of building from source |
| `gsl` | ✓ | Enable GSL backend for `BazinFit`/`VillarFit` (requires `libgsl-dev`) |
| `mkl` | | Intel MKL FFTW backend for the fast periodogram (strongly recommended for Intel CPUs) |
| `mimalloc` | ✓ | mimalloc memory allocator — up to 2× speedup for cheap features |

## Architecture

The package has a layered import structure:

1. `light_curve.light_curve_py` — pure-Python experimental implementations
2. `light_curve.light_curve_ext` — compiled Rust extension (via PyO3/Maturin)
3. `light_curve.__init__` — imports Python implementations first, then overrides with faster Rust equivalents

This means every extractor has a Python fallback, but the Rust version takes precedence at runtime.

**Rust source layout** (inside `light-curve/src/`):

| File | Contents |
|------|---------|
| `features.rs` | All 40+ feature extractors, PyO3 class definitions |
| `dmdt.rs` | dm-dt map implementation |
| `ln_prior.rs` | Log-prior helpers for MCMC fitting |
| `lib.rs` | Module exports and top-level PyO3 wiring |

## Adding a new feature extractor

1. Implement the feature in the upstream [`light-curve-feature`](https://github.com/light-curve/light-curve-feature) Rust crate.
2. Add PyO3 bindings in `src/features.rs` and export in `src/lib.rs`.
3. Optionally add a pure-Python experimental version in `light_curve/light_curve_py/features/`.
4. Add tests in `tests/light_curve_ext/test_feature.py`.
5. Add benchmarks in `tests/test_w_bench.py`.
6. Update `CHANGELOG.md`.

## Updating the documentation

Docs live in `docs/` on the `main` branch. To preview locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
cd light-curve && maturin develop --extras=full --group=docs -q && cd ..
mkdocs serve   # live preview at http://127.0.0.1:8000
```

**When adding a new feature extractor**, update the docs:

1. **Feature table** — `docs/features/index.md`: add a row to the appropriate category table (manually maintained).
2. **API page** — `docs/features/api/<category>.md`: add a `:::` autodoc entry:
   ```markdown
   ::: light_curve.NewFeatureName
       options:
         heading_level: 3
   ```
3. Commit and push; CI deploys automatically to the `dev` version of the site on merge to `main`.

The API reference prose (parameter descriptions, equations) is read from the Python
docstrings, so updating those in the source is enough — no manual copy-paste needed.

## Branding

Logo and brand assets live in a separate repository,
[`light-curve/branding`](https://github.com/light-curve/branding), together with the
Illustrator sources and the scripts that derive some of the SVGs. Do not edit the copies
under `docs/assets/logo/` — change them upstream and copy the result across, so the two stay
byte-identical.

Only what the site actually renders is vendored here:

| File | Used by |
|---|---|
| `mark-light.svg`, `mark-dark.svg` | header and drawer logo |
| `mark-adaptive.svg` | favicon |
| `mark-wide-light.svg`, `mark-wide-dark.svg` | landing page hero |

Keep it that way. MkDocs copies **everything** under `docs/` into the built site verbatim,
so an Illustrator file or a font archive dropped in here ships to visitors.

Each asset comes in a light-background and a dark-background variant, differing only in the
purple. Which one to show is chosen in three different ways, because the three contexts
offer different hooks:

- **Header and drawer** — `overrides/partials/logo.html` emits both, and
  `docs/stylesheets/extra.css` shows the one matching the active scheme. Material's palette
  toggle is independent of the reader's OS setting, so the switch has to follow the toggle.
- **Favicon** — `mark-adaptive.svg` carries both purples and picks between them with an
  internal `prefers-color-scheme` block. Nothing on the page can style a favicon, so this is
  the only option, and it follows the OS rather than the toggle.
- **README** — a `<picture>` with a `prefers-color-scheme` source, which GitHub resolves
  against its own theme setting.

The palette is defined in `extra.css`. Style with the role tokens `--lc-fg` and
`--lc-fg-accent`, not with the raw `--lc-purple` / `--lc-magenta` / `--lc-orange` /
`--lc-amber` — no single brand colour is legible in both schemes, so the roles swap between
them and feed Material's own link and accent variables. See the comments in `extra.css`.

### README images

`light-curve/README.md` is also the PyPI long description, so its image URLs must be
absolute, and they point at the branding repository. Two constraints are easy to trip over:

- PyPI sanitises the HTML with `readme_renderer` and **drops `<source>`**, keeping the
  `<img>`. The light-background variant therefore has to be the `<img>` fallback, since
  PyPI renders on a white page.
- Images must be served with an image content type. `raw.githubusercontent.com` serves SVG
  as `image/svg+xml`, so it works; not every host does.

To check a README change before pushing, render it the way PyPI will:

```bash
uvx --from 'readme_renderer[md]' python -c "
import readme_renderer.markdown, sys
print(readme_renderer.markdown.render(open('light-curve/README.md').read(), stream=sys.stderr))
"
```

## Docs preview CI

Every pull request gets an automatic docs preview at `https://light-curve.snad.space/pr<N>/`, posted as a comment by the bot.

The preview is built by the **Docs Preview** job in `docs.yml`, which uses a `pull_request_target` trigger so it has write access even for PRs from forks. It explicitly checks out the PR head commit to build the actual PR content.

Stale previews (from merged or closed PRs) are removed the next time anything is pushed to `main`.

## CI secrets

The test workflow uses three repository secrets. Jobs and steps that require them are
automatically skipped when the secret is absent, so CI still runs lint, build, and core tests
on fork pull requests.

| Secret | Used by | Purpose |
|--------|---------|---------|
| `HF_TOKEN` | `prepare-hf-models`, test jobs, `coverage`, `slow-tests` | Raises the HuggingFace API rate limit for model downloads (models are public; without the token the jobs still run at the unauthenticated limit) |
| `CODECOV_TOKEN` | `coverage` | Uploads the coverage report to Codecov (skipped when absent) |
| `CODSPEED_TOKEN` | `benchmarks` | Submits benchmark results to CodSpeed (whole job skipped when absent) |

Maintainers of the upstream repository configure these at **repo Settings → Secrets and variables → Actions**.
Fork contributors can add the same secrets to their own fork if they want full CI coverage.
To disable CI entirely on a fork, go to **Settings → Actions → General** and select **Disable actions**.

## Free-threaded Python

Free-threaded CPython (PEP 703) is supported from **Python 3.14t** onward.
Python 3.13t is not supported. To build for free-threaded Python, use
`--group dev-free-threading` instead of `--group dev` when installing dependencies
(avoids packages that don't yet support free threading: iminuit, polars, cesium).

Pre-built free-threaded wheels are not yet published to PyPI.

## Publishing a release

1. Create a `release-vX.Y.Z` branch from `main`.
2. Update `CHANGELOG.md` with the new version and date.
3. Update the version in `Cargo.toml` (the Python package version is read from there), then run
   `cargo update` to update `Cargo.lock`.
4. Commit, push the branch, and open a PR into `main`.
5. Tag the branch HEAD and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. CI (`publish.yml`) builds wheels via cibuildwheel and uploads to PyPI.
7. Once the release appears on PyPI, merge the PR into `main`.
8. Create a GitHub release from the tag (copy the relevant `CHANGELOG.md` section as the release notes).
