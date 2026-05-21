# Light-curve embeddings

`light_curve.embed` provides pretrained neural-network models that map raw photometric
time series to dense fixed-length vectors for downstream ML tasks: classification,
anomaly detection, and similarity search.

## Requirements

Install `light-curve` together with its ML embedding dependencies so all versions
resolve jointly:

```sh
pip install 'light-curve[full]' onnxruntime huggingface_hub   # CPU
pip install 'light-curve[full]' onnxruntime-gpu huggingface_hub  # NVIDIA GPU
```

`onnxruntime` is intentionally not bundled in the `light-curve[full]` extra — pick
the variant that matches your hardware. See the [onnxruntime install guide](https://onnxruntime.ai/docs/install/) for other providers.

If you already have the ONNX model file locally, `huggingface_hub` is not required.

## Available models

| Model | Bands | Input | Embedding dim | Pretrained on |
|-------|-------|-------|---------------|---------------|
| `Astromer1` | single | time, mag | 256 | MACHO R-band |
| `Astromer2` | single (or per-band) | time, mag | 256 | MACHO (1.5 M light curves) |
| `ATCAT` | 6 (ugrizY jointly) | time, flux, flux\_err, band index | 384 | ELAsTiCC |

## Single-band: Astromer2

[Astromer2](https://ui.adsabs.harvard.edu/abs/2026A%26A...707A.170D/abstract) accepts
irregularly-sampled `(time, mag)` pairs and returns 256-dimensional embeddings:

```python
import numpy as np
from light_curve.embed import Astromer2

model = Astromer2.from_hf(output="mean")

rng = np.random.default_rng(0)
time = np.sort(rng.uniform(0, 500, 120)).astype(np.float64)
mag  = rng.normal(15, 0.5, 120).astype(np.float64)

embedding = model(time, mag)
print(embedding.shape)  # (1, 1, 1, 256)
# squeeze to (256,) for a single object
vec = embedding.squeeze()
```

For multi-band data, pass `bands=["g", "r"]` to get one embedding per band:

```python
model = Astromer2.from_hf(output="mean", bands=["g", "r"])
embedding = model(time, mag, band=band)
print(embedding.shape)  # (2, 1, 1, 256)
```

## Multi-band: ATCAT

[ATCAT](https://ui.adsabs.harvard.edu/abs/2025arXiv251100614T/abstract) processes all six
LSST ugrizY bands jointly and returns 384-dimensional embeddings.
Inputs are flux (AB, zero-point 31.4 by default), flux error, time, and integer band index
(u=0, g=1, r=2, i=3, z=4, Y=5):

```python
from light_curve.embed import ATCAT

model = ATCAT.from_hf(output="last")

rng = np.random.default_rng(2)
n = 120
time     = np.sort(rng.uniform(0, 500, n)).astype(np.float32)
flux     = rng.normal(100, 10, n).astype(np.float32)
flux_err = np.full(n, 5.0, dtype=np.float32)
band     = np.array([i % 6 for i in range(n)])  # ugrizY → 0–5

embedding = model(time, flux, flux_err, band)
print(embedding.shape)  # (1, 1, 1, 384)
```

Set `mag_zp=27.5` for ELAsTiCC/SNANA FITS data, or `mag_zp=8.9` for Jy.

## GPU and alternative runtimes

Pass `ort_session_kwargs` to select an execution provider:

```python
model = Astromer2.from_hf(
    output="mean",
    ort_session_kwargs={"providers": ["CUDAExecutionProvider"]},
)
```

See the [onnxruntime install guide](https://onnxruntime.ai/docs/install/) for provider options.

See the [API reference](api.md) for full signatures and reduction strategies.
