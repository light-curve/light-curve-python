# onnxruntime tips

Practical configuration tweaks for running `light_curve.embed` models on HPC clusters, science platforms, and GPU nodes.

## Shared environments (science platforms and HPC clusters)

By default, onnxruntime grabs every CPU it can see and attempts to set thread
affinity accordingly. On shared nodes this can cause two problems: over-subscribing
your allocation, and outright failure when the host enforces strict CPU access control.
In the latter case you will see errors like:

```
pthread_setaffinity_np failed for thread: 8151, index: 18, mask: {73, }, error code: 22
error msg: Invalid argument. Specify the number of threads explicitly so the affinity is not set.
```

Fix both by setting thread counts explicitly via `SessionOptions`:

```python
import onnxruntime as ort
from light_curve.embed import Astromer2

so = ort.SessionOptions()
so.intra_op_num_threads = 4  # threads within a single op (e.g. matrix multiply)
so.inter_op_num_threads = 1  # threads across independent ops

model = Astromer2.from_hf(output="mean", ort_session_kwargs={"sess_options": so})
```

Set `intra_op_num_threads` to the number of CPUs in your job allocation
(e.g. from `$SLURM_CPUS_PER_TASK`).
`inter_op_num_threads = 1` is a safe default unless you have many independent model branches.

## GPU acceleration

### Enabling the CUDA provider

Install `onnxruntime-gpu` instead of `onnxruntime`, then request the CUDA provider:

```python
from light_curve.embed import Astromer2

model = Astromer2.from_hf(
    output="mean",
    ort_session_kwargs={"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
)
```

Always include `"CPUExecutionProvider"` as a fallback — onnxruntime will use it for any
operations the CUDA provider does not support.

### CUDA version compatibility

`onnxruntime-gpu` from PyPI is built against a specific CUDA version.
If the installed CUDA libraries do not match, you will see runtime errors like
`libcudnn.so.X: cannot open shared object file`.

The easiest way to get a compatible stack is to install everything through
**[pixi](https://pixi.sh)**, which uses conda-forge packages.
conda-forge's `onnxruntime-gpu` carries a `__cuda` virtual-package constraint
that lets the solver automatically pick a version compatible with your GPU driver —
no manual CUDA version selection needed:

```toml title="pixi.toml"
[workspace]
name = "my-project"
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = ">=3.10"
onnxruntime-gpu = "*"   # conda-forge resolves CUDA compatibility automatically
huggingface_hub = "*"
```

Run `nvidia-smi` to confirm the maximum CUDA version your driver supports
(shown in the top-right corner as *CUDA Version*).
If the driver is too old for the latest `onnxruntime-gpu`, pin to an older release:

```toml
onnxruntime-gpu = ">=1.18,<1.20"
```

### Other providers

onnxruntime supports additional hardware via execution providers:
DirectML (Windows/AMD), CoreML (Apple Silicon), OpenVINO (Intel), TensorRT, and others.
See the [onnxruntime execution providers docs](https://onnxruntime.ai/docs/execution-providers/)
for installation and provider names to pass in `providers`.
