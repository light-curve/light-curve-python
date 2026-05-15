# Light-curve embeddings

`light_curve.embed` provides pre-trained neural-network models for embedding light curves
into fixed-size vectors, useful for similarity search and classification.

```python
from light_curve.embed import Astromer2, EmbeddingSession

session = EmbeddingSession(Astromer2())
embeddings = session(light_curves)  # shape (N, 256)
```

See the [API reference](api.md) for all available models and reduction strategies.
