# RunPod experiment environment

The project's data generation, SFT, and local-model evaluation workflows
were developed and run with this configuration:

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA A40 |
| VRAM | 48 GB |
| Container image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Minimum host CUDA | 13.0 |
| NVIDIA driver | 580 series or newer |
| PyTorch | 2.13 with CUDA 13 |
| vLLM | 0.28.0 |
| Container storage | At least 60 GB |
| Persistent storage | RunPod volume mounted at `/workspace` |

The CUDA version in the container image name describes its preinstalled
user-space stack; the RunPod host supplies the NVIDIA driver. CUDA 13 support
is required by the verified PyTorch and vLLM combination.

The first Gemma 4 vLLM launch can spend several minutes compiling kernels and
capturing CUDA graphs. The evaluation configuration therefore uses a
600-second startup timeout, and the vLLM compilation cache should be preserved
between runs.
