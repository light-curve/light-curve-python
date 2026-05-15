# Documentation Website Plan

## Technology stack

### Framework: MkDocs + Material for MkDocs

**Recommended: MkDocs with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.**

Rationale:
- Modern, fresh design that works beautifully out of the box — no custom CSS needed for 95% of the desired look
- Code blocks: syntax highlighting, one-click copy button, and line annotations are built-in features of Material
- GitHub "Edit this page" links: built-in, one config option (`edit_uri`)
- Clickable cross-references between classes and functions: provided by **mkdocstrings** (see below)
- Full-text search: built-in, fast, client-side
- Responsive layout for the landing page grid
- Markdown-based — contributors do not need to learn RST
- Widely used in the scientific Python ecosystem (xarray, zarr, astropy subprojects)

Alternatives considered:
- **Sphinx + PyData theme**: powerful but RST-heavy and slower to author. Better for cross-referencing across projects (Intersphinx). Prefer MkDocs for authoring speed and design.
- **Jupyter Book**: excellent for notebook-centric sites but heavier toolchain.
- **Quarto**: modern and notebook-friendly but adds a non-Python dependency.

### API documentation: mkdocstrings

[mkdocstrings](https://mkdocstrings.github.io/) extracts docstrings and type signatures
at build time and renders them as linked, searchable API pages. Usage in Markdown:

```markdown
::: light_curve.Amplitude
::: light_curve.Periodogram
```

This gives clickable type annotations, inheritance diagrams, and cross-links between
classes automatically. Because `light_curve` is a Rust extension, the docstrings live
in `src/features.rs` (exposed via PyO3) and in the Python stubs. mkdocstrings works
with the installed package (not the source), so the docs build installs the wheel from
PyPI and introspects it at runtime — no Rust compilation needed during docs build.

### Tutorials: mkdocs-jupyter

[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) converts `.ipynb`
notebooks to docs pages during build. Tutorials are authored as notebooks (runnable
locally, rendered statically in docs), stored under `docs/tutorials/`.

### Hosting: Read the Docs

**Decided: [Read the Docs](https://readthedocs.org/).**

Advantages over GitHub Pages:
- PR preview builds (each PR gets its own docs URL)
- Versioned docs (`/stable/`, `/latest/`, `/v0.9.x/`)
- Built-in search across versions
- No separate GitHub Actions workflow needed for deployment

Setup: connect the repo at readthedocs.org, add `.readthedocs.yaml` config file.
The build installs `light-curve[full]` from PyPI (no Rust compilation needed).

---

## Site structure

```
docs/
├── index.md                      # Landing page (see §Landing page)
├── installation.md               # Detailed install instructions
│
├── features/                     # Hand-crafted feature extractors
│   ├── index.md                  # Overview + feature table
│   ├── tutorial-basics.ipynb     # Getting started: Extractor, names, many()
│   ├── tutorial-periodogram.ipynb
│   ├── tutorial-nonlinear-fit.ipynb   # BazinFit, VillarFit, Rainbow
│   ├── tutorial-binning.ipynb
│   ├── tutorial-batch.ipynb           # .many(), Arrow/Polars/nested-pandas
│   └── api.md                    # Full API reference (mkdocstrings)
│
├── embed/                        # Machine learning embeddings
│   ├── index.md                  # Overview: Astromer, ATCAT, ONNX model system
│   ├── tutorial-similarity.ipynb # Nearest-neighbour search on ZTF DR23
│   ├── tutorial-classification.ipynb
│   └── api.md
│
├── dmdt/                         # dm-dt maps
│   ├── index.md                  # Overview + visual explanation
│   ├── tutorial-cnn.ipynb        # Training a CNN on dm-dt maps
│   └── api.md
│
└── developer/
    ├── contributing.md
    ├── changelog.md              # Auto-generated from CHANGELOG.md
    └── citation.md
```

---

## Landing page

The landing page (`docs/index.md`) has four sections:

### 1. Hero

Full-width banner:
- Package name + one-sentence description
- Two buttons: **Get started** (→ installation) and **View on GitHub**
- Background: subtle animated star-field or dark gradient (pure CSS)

### 2. Install snippet

A single centered code block (one-click copy):

```sh
pip install 'light-curve[full]'
```

### 3. Three-column feature blocks

Three cards, displayed side-by-side on wide screens, stacked on mobile.
Each card has:
- An **animated SVG illustration** that plays on hover (see below)
- A short title and two-line description
- A "Learn more →" link

| Card | Title | Static state | Hover animation |
|------|-------|--------------|-----------------|
| A | Hand-crafted features | A simple light curve plot | The curve fades, two labelled quantities appear: amplitude (↕ bracket) and a period marker (↔ bracket). CSS keyframe, ~0.6 s ease |
| B | ML embeddings | A neural-network layer diagram (boxes and arrows) | A light curve enters left, flows through layers, a vector `[0.3, -1.2, …]` emerges right. SVG stroke-dashoffset animation |
| C | dm-dt maps | A 2D heat-map grid (static colour gradient) | The light curve appears on the left, dots scatter across the dm-dt grid one by one (staggered SVG circle animation) |

All animations are pure CSS + inline SVG — no JavaScript libraries, no external CDN
dependencies. They degrade gracefully (static image) when `prefers-reduced-motion` is
set.

### 4. Quick-start code block

A single runnable example (same as the README Quick start), shown with full syntax
highlighting.

---

## Feature section details

### Overview page (`features/index.md`)

- One-paragraph introduction
- A searchable table of all extractors: Name | Description | Output dimension | Tags (periodogram, fitting, statistical…)
- The table is generated at docs-build time by a small Python script that imports
  `light_curve` and iterates over all objects that have `.names`, emitting a Markdown
  table. This ensures the table is always in sync with the installed version.

### API reference (`features/api.md`)

mkdocstrings renders every public class in `light_curve` that has a `.names` attribute,
grouped by category (statistical, periodogram-based, fitting, multiband, meta).
Each entry shows:
- Class signature with linked type annotations
- Docstring (parameters, returns, notes, references)
- Inherited members (e.g., all features share `.names`, `.descriptions`, `__call__`)

Because all features share the same interface, a single "Feature interface" page
documents the shared methods once, and individual feature pages cross-link to it.
This eliminates the duplication mentioned in the requirements.

### Tutorials

| Notebook | Content |
|----------|---------|
| `tutorial-basics.ipynb` | Creating features, `Extractor`, `.names`, `.descriptions`, single call, error handling |
| `tutorial-periodogram.ipynb` | `Periodogram`, `PeriodogramPeaksByInverseDistance`, spectrum features, Lomb-Scargle internals |
| `tutorial-nonlinear-fit.ipynb` | `BazinFit`, `VillarFit`, `RainbowFit`, iminuit backend, convergence diagnostics |
| `tutorial-binning.ipynb` | `Bins`, choosing bin width, multi-band binning |
| `tutorial-batch.ipynb` | `.many()`, Arrow/Polars/pandas/nested-pandas input, performance comparison |

---

## Embed section details

### Overview page

- How the embedding system works: ONNX models, `EmbeddingSession`, `SingleBandModel`
- Model catalogue table: Model | Bands | Input shape | Embedding dim | Paper

### Tutorials

| Notebook | Content |
|----------|---------|
| `tutorial-similarity.ipynb` | Load Astromer2, embed ZTF DR23 light curves, build a FAISS/sklearn KNN index, query by object ID |
| `tutorial-classification.ipynb` | Embed a labelled dataset, train a logistic-regression classifier, compare to raw features |

---

## dm-dt section details

### Overview page

- What a dm-dt map is, with a visual example (static image)
- Link to the Makarov+2021 paper

### Tutorial

| Notebook | Content |
|----------|---------|
| `tutorial-cnn.ipynb` | Generate dm-dt maps for a ZTF sample, train a simple CNN (PyTorch/Keras), evaluate |

---

## Technical implementation

### `mkdocs.yml` (top-level config skeleton)

```yaml
site_name: light-curve
site_url: https://light-curve.github.io/light-curve-python/
repo_url: https://github.com/light-curve/light-curve-python
repo_name: light-curve/light-curve-python
edit_uri: edit/main/docs/

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: deep-orange
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: deep-orange
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs          # top-level nav as tabs
    - navigation.sections
    - navigation.top           # back-to-top button
    - content.code.copy        # copy button on all code blocks
    - content.action.edit      # "Edit this page" GitHub link

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            show_bases: true
            docstring_style: google
  - mkdocs-jupyter:
      execute: false           # notebooks are pre-executed and stored with output

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - attr_list
  - md_in_html                # needed for the landing page cards
```

### Docs dependency group (added to `pyproject.toml`)

```toml
[dependency-groups]
docs = [
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
    "mkdocs-jupyter>=0.24",
    "light-curve[full]",      # install from PyPI; no Rust build needed
]
```

### Feature table generator (`docs/scripts/gen_feature_table.py`)

A small script run at docs-build time (via mkdocs `hooks:`) that:

```python
import light_curve as lc, inspect

CATEGORIES = { ... }  # map class → category

rows = []
for name in sorted(dir(lc)):
    obj = getattr(lc, name)
    if hasattr(obj, "names"):
        rows.append(f"| [{name}](api.md#{name}) | {obj.__doc__.splitlines()[0]} | {len(obj().names)} | ... |")

Path("docs/features/table.md").write_text("\n".join(rows))
```

This guarantees the feature table is never out of date.

### Read the Docs config (`.readthedocs.yaml`)

```yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.12"
  commands:
    - pip install uv
    - uv sync --group docs
    - uv run mkdocs build --strict --site-dir $READTHEDOCS_OUTPUT/html

mkdocs:
  configuration: mkdocs.yml
```

The build installs `light-curve[full]` from PyPI (no Rust toolchain required),
completing in ~2 minutes. Read the Docs triggers builds on every push to `main`
and on every PR.

---

## Design decisions / open questions

1. **Hosting — GitHub Pages vs Read the Docs**: GitHub Pages is simpler to set up
   (proposed above). Read the Docs adds PR preview links and versioned docs
   (`/v0.9.x/` vs `/latest/`). Decision deferred to maintainer.

2. **Dark mode**: Material's built-in palette toggle is included in the config above.
   The landing-page SVG animations need a dark-mode variant (stroke colours inverted).

3. **Notebook execution**: Notebooks are stored pre-executed (outputs committed) so
   the docs build is fast and does not require HuggingFace access for the embed
   tutorials. A separate nightly CI job can re-execute and push updated notebooks.

4. **Rust docstrings and `.pyi` stubs**: PyO3 exposes docstrings written in Rust source.
   mkdocstrings introspects the installed wheel, so these appear. However, no `.pyi`
   stub files exist yet, so parameter type annotations will be missing or incomplete in
   the API docs. **Adding `.pyi` stubs is a prerequisite for full API doc quality** and
   should be a parallel task (tracked separately). Until stubs exist, we can use
   `show_signature_annotations: false` in mkdocstrings config to avoid showing
   incorrect types.

5. **Custom domain**: The site will initially live at
   `https://light-curve.github.io/light-curve-python/`. A custom domain
   (`docs.light-curve.space` or similar) can be added later with a CNAME record.

---

## Phased implementation

| Phase | Scope | Effort |
|-------|-------|--------|
| 0 | *(done)* CI disabled on `documentation` branch (`test.yml` uses `branches-ignore`) | — |
| 1 | mkdocs.yml, `.readthedocs.yaml`, theme, skeleton pages, install page, API reference for features | ~1 day |
| 2 | Landing page with animated SVG cards, dark-mode polish | ~1 day |
| 3 | Features tutorials (5 notebooks) | ~2 days |
| 4 | Embed section + 2 tutorials | ~1 day |
| 5 | dm-dt section + CNN tutorial | ~1 day |
| 6 | Feature table generator script, changelog page | ~0.5 day |
| 7 | **Re-enable CI**: remove `branches-ignore` from `.github/workflows/test.yml`; add a separate lightweight `docs-build` CI job that runs `mkdocs build --strict` on every PR to catch broken links and missing references | ~0.5 day |
