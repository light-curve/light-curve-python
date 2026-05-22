---
hide:
  - navigation
---

# Installation

## Quick install

=== "pip"

    ```sh
    pip install 'light-curve[full]'
    ```

=== "conda"

    ```sh
    conda install conda-forge::light-curve-python
    ```

The `[full]` extra adds two packages needed for parametric fitting features:
**[iminuit](https://iminuit.readthedocs.io/)** (MCMC optimisation) and
**[scipy](https://scipy.org/)** (least-squares solver).
Without them, `BazinFit`, `LinexpFit`, and `VillarFit` still work with the compiled Ceres/GSL backends.

Minimum Python version: **3.10**. Binary wheels are provided for all major platforms — no Rust toolchain required.

## Platform support

| Arch \ OS | Linux glibc 2.17+ | Linux musl 1.2+ | macOS | Windows |
|-----------|-------------------|-----------------|-------|---------|
| **x86-64** | PyPI (MKL), conda | PyPI (MKL) | PyPI (macOS 15+), conda | PyPI, conda |
| **aarch64** | PyPI | PyPI | PyPI (macOS 14+), conda | not tested |
| **i686 / ppc64le** | build from source | — | — | — |

Notes:

- **MKL**: Linux x86-64 PyPI wheels use Intel MKL for a significantly faster periodogram. Conda and other platforms use the pure-Rust [RustFFT](https://crates.io/crates/rustfft) backend.
- **Windows conda**: Excludes GSL, so the `"lmsder"` fitting backend is unavailable. PyPI wheels on Windows include GSL.
- **macOS**: Requires macOS 14+ (aarch64) or 15+ (x86-64). Open an [issue](https://github.com/light-curve/light-curve-python/issues) if you need support for an older version.

## Embeddings (`light_curve.embed`)

The embedding models use [ONNX Runtime](https://onnxruntime.ai) for inference. You must install the
runtime yourself because the right package depends on your hardware:

| Hardware | Package |
|----------|---------|
| CPU | `pip install onnxruntime` |
| NVIDIA GPU | `pip install onnxruntime-gpu` |
| Other | see [onnxruntime.ai/docs/install](https://onnxruntime.ai/docs/install/) |

Models are stored on [HuggingFace Hub](https://huggingface.co/light-curve) and downloaded
automatically on first use. To enable automatic downloads, install:

```sh
pip install huggingface_hub
```

If you already have the ONNX file locally, `huggingface_hub` is not needed.

## Build from source

Building requires a **Rust toolchain** (≥ 1.85, install via [rustup](https://rustup.rs/))
and **CMake** (for the Ceres solver). GSL is optional but enables the `"lmsder"` fitting backend.

```sh
git clone https://github.com/light-curve/light-curve-python
cd light-curve-python/light-curve
pip install maturin
maturin develop --extras=dev      # dev build with test deps
maturin develop --release         # optimised build
```

See [Contributing](developer/contributing.md) for the full developer guide.
