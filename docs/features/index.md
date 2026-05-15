# Feature extractors

`light_curve` provides 50+ hand-crafted feature extractors for astrophysical light curves.
All features share a common interface: they are callable objects with `.names` and `.descriptions` attributes.

```python
import light_curve as lc

extractor = lc.Extractor(lc.Amplitude(), lc.Periodogram(peaks=1))
result = extractor(t, m, err)
```

See the [API reference](api.md) for the full list of available extractors.
