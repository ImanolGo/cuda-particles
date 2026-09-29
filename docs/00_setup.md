# 00 — Setting up CUDA

You need three separate things. Confusing them is the usual reason a "CUDA
install" goes wrong:

| Piece | What it is | You get it from |
|---|---|---|
| **GPU driver** | Talks to the card. Provides `nvidia-smi` and `/dev/nvidia*`. | your distro (`nvidia-open`, `nvidia`…) |
| **CUDA Toolkit** | `nvcc`, headers, libraries. This is what *compiles* your code. | `cuda` package / NVIDIA installer |
| **Runtime** | `libcudart`, ships with the driver **and** the toolkit. | comes along for free |

The driver exposes a maximum CUDA version; the toolkit must be **≤** that.
On your machine:

```
$ nvidia-smi
| NVIDIA-SMI 615.71.09      Driver Version: 615.71.09
| CUDA Version: 13.4          <-- the driver can run up to CUDA 13.4
```

So a CUDA 13.x toolkit is a perfect fit.

---

## 1. Install the toolkit

### Arch / CachyOS / Manjaro

```bash
sudo pacman -S cuda          # pulls nvcc into /opt/cuda  (~4 GB)
```

Then make it reachable from your shell:

```bash
# add to ~/.bashrc (or ~/.zshrc) and re-open the terminal
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
```

The `Makefile` already falls back to `/opt/cuda/bin/nvcc` if `nvcc` is not on
`PATH`, so setting the env vars is optional for this repo.

Optional but lovely once you are comfortable:

```bash
sudo pacman -S nsight-compute nsight-systems    # `ncu` and `nsys` profilers
```

### Other systems

* **Ubuntu / Debian** — `sudo apt install nvidia-cuda-toolkit` (often old) or
  use NVIDIA's official `.deb` repo for the newest build.
* **Fedora** — `sudo dnf install cuda-toolkit`.
* **Windows** — the CUDA Toolkit installer from NVIDIA; build from a
  "x64 Native Tools" prompt. Everything in this repo works, but the window
  examples use GLFW which is a Linux-flavoured setup here.

---

## 2. Verify

```bash
make doctor          # prints nvcc + GPU
nvcc --version       # compiler + toolkit version
nvidia-smi           # driver + GPU
```

Then the real test:

```bash
make run-01_hello
```

You should see your card's name, compute capability and a grid of threads
printing their global ids. If that runs, your toolchain is healthy.

---

## 3. The one number that matters: compute capability

Every NVIDIA GPU has a **compute capability** (`sm_XY`) that describes which
instructions it understands. In CUDA 9+ you no longer need to know it by hand:

```bash
nvcc -arch=native ...        # nvcc asks the GPU and compiles for it
```

Your **GTX 1650 Mobile = Turing = `sm_75`**. If you ever compile on a build
machine without the GPU present (CI, containers), pass it explicitly:

```bash
make ARCH=sm_75
```

`ARCH` is a plain Makefile variable, so that is all you type. You can also
target several architectures in one binary by extending `NVCCFLAGS` with e.g.
`-gencode arch=compute_75,code=sm_75` — worth knowing, never required here.

---

## 4. What `nvcc` actually does

`nvcc` is not a GPU compiler in the usual sense — it is a *driver* that runs
two compilers:

```
   your.cu
      │
      ├─ host code  ──►  gcc/g++        ──► normal CPU object file
      │
      └─ device code ─►  cicc/ptxas     ──► PTX / SASS embedded in the binary
```

The two halves are stitched together into one executable. This is why:

* host-compiler problems (e.g. "unsupported GNU version") are *nvcc* problems,
* `printf` in a kernel exists at all (the runtime patches it over),
* and a binary built for `sm_75` will not run on an older card.

---

## 5. Troubleshooting

**`nvcc: command not found`** — install the `cuda` package, then check
`/opt/cuda/bin/nvcc` exists. The Makefile will find it either way.

**`unsupported GNU version! gcc 16 is not supported`** — your GCC is newer than
what this nvcc was tested against. Rebuild with the check disabled:

```bash
make UNSUPPORTED=1
```

This usually just works. If it does not, install an older host compiler and
point nvcc at it: `make NVCC="nvcc -ccbin g++-14"` (Arch also has `gcc14` in
the AUR / `gcc14` package).

**`error: unsupported gpu architecture 'compute_XX'`** — you asked for an arch
your toolkit does not know. Drop `ARCH=native` only if you are cross-compiling.

**The window opens then closes instantly / no display** — the particle examples
need a desktop session. On Wayland they default to XWayland for CUDA/OpenGL
interop reliability; see the *Troubleshooting* section of the main README.

**Everything compiles but output is zeros** — you forgot to copy the result
back, or your kernel silently failed. Wrap the call in `CUDA_CHECK(...)` from
`common/cuda_check.h` and add `CUDA_CHECK_KERNEL()` after the launch.
