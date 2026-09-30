# CUDA from Scratch — a particle playground

A step-by-step CUDA course for people who come from creative coding. Seven tiny
programs that build up the fundamentals, ending in an interactive GPU particle
system you stir with the mouse: hundreds of thousands of particles, colour from
velocity, trails, swirling.

<p align="center">
  <img src="docs/demo.webp" alt="The CUDA particle system running" width="720">
</p>

```
   examples/01  hello, GPU              →  kernels, threads, blocks
   examples/02  vector add              →  device memory, copying, errors
   examples/03  index math              →  thread-per-element, grid-stride loops
   examples/04  shared memory           →  tiles, __syncthreads, bank conflicts
   examples/05  timing & streams        →  events, overlap, pinned memory
   examples/06  CUDA ↔ OpenGL interop   →  the bridge to the final project
   particles/                           →  the final project
```

## Quick start

```bash
# 1. install the toolkit (Arch / CachyOS / Manjaro)
sudo pacman -S cuda
export PATH=/opt/cuda/bin:$PATH        # optional; the Makefile finds /opt/cuda

# 2. see what you have
make doctor

# 3. walk the path
make run-01_hello
make run-02_vector_add
make run-03_index_math
make run-04_shared_memory
make run-05_timing_streams
make run-06_interop                    # a window, a CUDA-animated triangle

# 4. the final project
make run-particles
```

`nvcc` not installed yet? → **[docs/00_setup.md](docs/00_setup.md)** covers the
whole setup, what `nvcc` actually does, and how to fix the usual failures.

## The final project

```bash
make run-particles
make run-particles ARGS="--count 1000000 --fade 0.05"
```

| input | effect |
|---|---|
| move the mouse | the cloud follows and swirls around the cursor |
| left button | strong attraction |
| right button | strong repulsion |
| `space` | pause / resume |
| `R` | re-seed the cloud |
| `+` / `-` | bigger / smaller particles |
| `up` / `down` | shorter / longer trails |
| `ESC` | quit |

Options: `--count N` (default 400 000), `--fade F` (trail fade per frame,
0–1; `0` leaves permanent trails), `--size P` (base point size in pixels).

### How it works

Three things happen every frame, and nothing is copied through the CPU:

```
   ┌─────────────────────────── GPU ───────────────────────────┐
   │                                                           │
   │  1. CUDA writes positions + colours                       │
   │        straight into the OpenGL vertex buffers            │
   │                                                           │
   │  2. a full-screen black triangle at alpha 0.09            │
   │        fades last frame toward black  →  motion trails     │
   │                                                           │
   │  3. OpenGL draws those buffers as GL_POINTS               │
   │        additive blending              →  glow              │
   │                                                           │
   └───────────────────────────────────────────────────────────┘
```

The whole exchange is five calls, the same ones you write by hand in
`examples/06_interop`:

```cpp
cudaGraphicsMapResources(2, resources, 0);          // let CUDA in
cudaGraphicsResourceGetMappedPointer(...);          // here is the device pointer
particles_update<<<grid, block>>>(...);             // simulate straight into it
cudaGraphicsUnmapResources(2, resources, 0);        // hand it back to OpenGL
glDrawArrays(GL_POINTS, 0, count);                  // draw
```

### The simulation

`particles/src/physics.cu` is the whole story: one thread per particle, no
shared memory, no synchronisation. Each particle feels

* an inverse-square pull toward the cursor, damped by a gaussian so only nearby
  particles react;
* a **swirl** — the same force rotated 90°, which is what makes it curl instead
  of collapsing;
* a pinch of random jitter, so the field never freezes;
* exponential damping toward rest;
* a soft bounce off the edges.

Colour comes from the *direction* of travel (so particles in the same flow
share a colour and you get ribbons) and brightness from speed. The palette is
three offset cosines — a one-line trick worth stealing for any project.

Every magic number sits at the top of the frame loop in
`particles/src/main.cu` (`attraction`, `swirl`, `damping`, `jitter`, `radius`).
Change one, rebuild, and watch what happens. That is the fastest way to build
intuition.

## Repository layout

```
cuda-particles/
├── Makefile                  build everything, or one thing
├── common/cuda_check.h       CUDA_CHECK — use it, always
├── docs/00_setup.md          installing the toolkit, troubleshooting
├── examples/
│   ├── 01_hello/main.cu
│   ├── 02_vector_add/main.cu
│   ├── 03_index_math/main.cu
│   ├── 04_shared_memory/main.cu
│   ├── 05_timing_streams/main.cu
│   └── 06_interop/main.cu
└── particles/
    ├── src/main.cu           window, GL state, frame loop
    ├── src/physics.cu        the kernels
    ├── src/particles.cuh     shared data layout
    └── src/shaders.h         GLSL
```

Each example is a single self-contained `.cu` file with heavy comments. The
final project is split across files the way a real one would be.

## Build options

```bash
make                       # everything
make 01_hello              # one example
make run-03_index_math     # build & run one example
make particles             # just the final project
make clean

make ARCH=sm_75            # compile for a specific GPU instead of `native`
make UNSUPPORTED=1         # if nvcc rejects your (newer) host GCC
make NVCC=/opt/cuda-12.6/bin/nvcc
```

`-arch=native` asks `nvcc` to detect your card at compile time, so you normally
never touch `ARCH`. Your GTX 1650 Mobile is Turing, `sm_75`.

## Troubleshooting

**`nvcc: command not found`** — install the `cuda` package. The Makefile falls
back to `/opt/cuda/bin/nvcc`, so `make doctor` should still find it.

**`unsupported GNU version! gcc 16 is not supported`** — `make UNSUPPORTED=1`.
See `docs/00_setup.md` for the details and for using an older host compiler.

**Windowed examples fail with "Cannot pair CUDA with this OpenGL context"** —
you are on a laptop with hybrid graphics (Optimus): the window was created on
the integrated GPU, so CUDA and OpenGL are on different devices and cannot
share buffers. Move the GL context to the NVIDIA GPU:

```bash
prime-run make run-particles
# or, equivalently:
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia make run-particles
```

`make run-X` already does this for you when `prime-run` (package
`nvidia-prime`) is installed. Override with `make run-particles RUN=`.

**Blank window (and the GPU check above passed)** — the particle examples
default to an **X11 (XWayland) GL context**, because GLX is the well-supported
path for CUDA/OpenGL interop on Linux. To try a native Wayland (EGL) context
instead:

```bash
CUDA_PARTICLES_PLATFORM=wayland make run-particles
CUDA_PARTICLES_PLATFORM=any     make run-particles
```

**Everything compiles but the output is all zeros** — a kernel failed silently.
Wrap calls in `CUDA_CHECK(...)` (see `common/cuda_check.h`) and add
`CUDA_CHECK_KERNEL()` after the launch; it will tell you what went wrong and
where.

**Sluggish at 400 000 particles** — lower `--count`, or check the FPS in the
title bar. The simulation is trivially parallel; if it is slow, it is almost
always the *fill rate* of the points, so try a smaller `--size`.

**`nvcc fatal : Unsupported gpu architecture 'compute_XX'`** — you asked for an
architecture this toolkit does not know. Drop `ARCH` and use `native`.

## Ideas to take it further

Ordered roughly by difficulty:

* Colour by position instead of velocity — one line in `update_kernel`.
* Give particles a finite life: recycle the dead ones at the cursor, and watch
  it turn into a fountain.
* Add a `mouse velocity` force so the cursor pushes particles like wind.
* Move the trail fade into an off-screen framebuffer (FBO) and ping-pong it, so
  the trails can be blurred or distorted.
* Sort particles for painter's algorithm, or render them as instanced quads
  instead of point sprites.
* Read `ncu` (Nsight Compute) output for `update_kernel` and find out whether it
  is memory- or compute-bound. Then make it faster.

## License

MIT — see [LICENSE](LICENSE).
