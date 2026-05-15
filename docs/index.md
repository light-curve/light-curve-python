# light-curve

High-performance time-series feature extraction for astrophysics.

```sh
pip install 'light-curve[full]'
```

## What's inside

<div class="grid cards" markdown>

-   **Hand-crafted features**

    50+ feature extractors for astrophysical light-curve characterisation — amplitude, period, colour, variability indices, and more.

    [:octicons-arrow-right-24: Features](features/index.md)

-   **ML embeddings**

    Embed light curves with pre-trained neural networks (Astromer, ATCAT) for similarity search and classification.

    [:octicons-arrow-right-24: Embeddings](embed/index.md)

-   **dm-dt maps**

    Transform light curves into 2D magnitude-difference vs time-difference maps for CNN-based classification.

    [:octicons-arrow-right-24: dm-dt maps](dmdt/index.md)

</div>

## Quick start

```python
import light_curve as lc
import numpy as np

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0, 100, 100))
m = 15.0 + 0.01 * t + rng.normal(0, 0.1, 100)
err = np.full(100, 0.1)

extractor = lc.Extractor(lc.Amplitude(), lc.BeyondNStd(nstd=1), lc.LinearFit())
result = extractor(t, m, err)
print(dict(zip(extractor.names, result)))
```
