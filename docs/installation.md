# Installation

## Quick install

=== "pip"

    ```sh
    pip install 'light-curve[full]'
    ```

=== "conda"

    ```sh
    conda install -c conda-forge light-curve-python
    ```

The `[full]` extra installs all optional Python dependencies for experimental features.
`light-curve-python` on conda-forge is an alias for the full package.

Minimum Python version: **3.10**.
No Rust toolchain or compilation is needed — binary wheels are available for all major platforms.

## Extras

| Extra | What it adds |
|-------|-------------|
| *(none)* | Core feature extractors only |
| `full` | Experimental features (`RainbowFit`, multiband features, detection-based features) |

## Platform support

| Arch \ OS | Linux glibc 2.17+ | Linux musl 1.2+ | macOS | Windows |
|-----------|-------------------|-----------------|-------|---------|
| **x86-64** | PyPI (MKL), conda | PyPI (MKL) | PyPI (macOS 15+), conda | PyPI, conda |
| **aarch64** | PyPI | PyPI | PyPI (macOS 14+), conda | not tested |
| **i686 / ppc64le** | build from source | — | — | — |

- **MKL**: Intel MKL-accelerated periodogram on Linux x86-64 (PyPI and conda).
- **Windows conda**: Excludes GSL, so the `"lmsder"` solver for `BazinFit`/`VillarFit` is unavailable. PyPI wheels on Windows include full solver support.
- **macOS**: Wheels require macOS 14+ (aarch64) or 15+ (x86-64). Open an issue if you need older macOS support.

## Optional Python dependencies

Some use cases require additional packages:

| Use case | Package |
|---------|---------|
| `.many()` with nested-pandas input | `pip install nested-pandas` |
| `.many()` with PyArrow input | `pip install pyarrow` |
| `.many()` with Polars input | `pip install polars` |
| Embeddings (`light_curve.EmbeddingSession`) | `pip install onnxruntime huggingface_hub` |

## Build from source

Building requires a **Rust toolchain** (≥ 1.85, install via [rustup](https://rustup.rs/))
and **CMake** (for the Ceres solver). GSL is optional but enables the `"lmsder"` solver.

```sh
git clone https://github.com/light-curve/light-curve-python
cd light-curve-python/light-curve
pip install maturin
maturin develop --extras=dev      # dev build with test deps
maturin develop --release         # optimised build
```

After Rust changes, re-run `maturin develop`. Python-only changes need no rebuild.

See the [Contributing](developer/contributing.md) page for the full developer guide.
