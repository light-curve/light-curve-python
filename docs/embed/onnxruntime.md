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
so.inter_op_num_threads = 1  # threads across independent ops in the graph

model = Astromer2.from_hf(output="mean", ort_session_kwargs={"sess_options": so})
```

**Both values must be set explicitly** — onnxruntime only skips the affinity call when a
thread count is given. Set their sum to the number of CPUs in your job allocation
(e.g. `$SLURM_CPUS_PER_TASK`). All models in `light_curve.embed` have parallel op
subgraphs (multi-head attention etc.), so both dimensions can benefit from more than one
thread; the example above is a reasonable starting point rather than a universal optimum.

## GPU acceleration

### Enabling the CUDA provider

Install `onnxruntime-gpu` instead of `onnxruntime`, then request the CUDA provider:

```python
from light_curve.embed import Astromer2

model = Astromer2.from_hf(
    output="mean",
    ort_session_kwargs={"providers": ["CUDAExecutionProvider"]},
)
```

To confirm the GPU is actually being used, run `nvidia-smi` while inference is running
and look for your Python process in the *Processes* table at the bottom.

### CUDA version compatibility

`onnxruntime-gpu` from PyPI is built against a specific CUDA version.
If the installed CUDA libraries do not match, you will see runtime errors like
`libcudnn.so.X: cannot open shared object file`.

The [CUDA Execution Provider requirements table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
lists the exact CUDA and cuDNN versions required by each `onnxruntime-gpu` release.

The easiest way to get a compatible stack is **[pixi](https://pixi.sh)**, which can
install the CUDA toolkit and cuDNN from the NVIDIA conda channel alongside
`onnxruntime-gpu` from PyPI:

```toml title="pixi.toml"
[workspace]
name = "my-project"
channels = ["conda-forge", "nvidia/label/cuda-12.6.3"]
platforms = ["linux-64"]

[dependencies]
python = ">=3.10"
cuda-toolkit = { version = "12.*", channel = "nvidia/label/cuda-12.6.3" }
cudnn = "9.*"
huggingface_hub = "*"

[pypi-dependencies]
onnxruntime-gpu = "*"
```

First run `nvidia-smi` and check the *CUDA Version* shown in the top-right corner —
that is the maximum version your driver supports.
Choose a `nvidia/label/cuda-X.Y.Z` channel that is at or below that version.
If `onnxruntime-gpu` from PyPI requires a different CUDA version than the one you
installed, pin it explicitly:

```toml
[pypi-dependencies]
onnxruntime-gpu = ">=1.18,<1.20"
```

### Other providers

onnxruntime supports additional hardware via execution providers:
DirectML (Windows/AMD), CoreML (Apple Silicon), OpenVINO (Intel), TensorRT, and others.
See the [onnxruntime execution providers docs](https://onnxruntime.ai/docs/execution-providers/)
for installation and provider names to pass in `providers`.
