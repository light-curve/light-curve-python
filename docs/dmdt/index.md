# dm-dt maps

A dm-dt map transforms a light curve into a 2D histogram of magnitude differences (dm)
vs time differences (dt), introduced by [Makarov et al. 2021](https://arxiv.org/abs/2104.05551).

```python
import light_curve as lc
import numpy as np

dmdt = lc.DmDt.from_borders(
    min_lgdt=0, max_lgdt=2, max_abs_dm=3,
    lgdt_size=32, dm_size=32,
    norm=["lgdt", "dm"],
)
map_ = dmdt.points(t, m)  # shape (32, 32)
```

See the [API reference](api.md).
