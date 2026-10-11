# Review: from CUDA fundamentals to a visual instrument

## What the project gets right

The intent is unusually good: teach CUDA to creative coders without making them
learn a renderer, game engine, or large framework first. Six single-file lessons
lead into a small, GPU-resident final project. Keep that progression.

The final simulation is an excellent first GPU workload: one thread owns one
particle, with coalesced arrays and no neighbour dependencies. Shared memory
is intentionally **not** used here—it would add synchronization without helping.
CUDA/OpenGL interop is the important payoff, not just a larger vector-add kernel.

## Issues fixed during this review

| Finding | Change |
|---|---|
| README advertised `ARGS`, but Make never forwarded it | run targets now pass `$(ARGS)` |
| Timing lesson read uninitialized `d_in` | initialize inputs before warming up |
| A finite checksum did not prove stream correctness | compare every result with serial output and a CPU FMA reference |
| Several lessons always returned success | checks now fail the process, making `make check` meaningful |
| Convenient sizes never exercised excess threads / edge tiles | use non-block-aligned vector, matrix, and transpose sizes |
| Transpose description mixed up reads and writes | correct the explanation and check both kernels exhaustively |
| Swapped back-buffer contents are not persistent history | explicit floating-point trail framebuffer |
| Reversed GLSL `smoothstep` edges are undefined | use ascending edges and invert the result |
| Trail length and brightness changed with display rate | time-based fade and light deposition |
| Pause kept depositing light; reset left old trails | freeze history while paused and clear on reset |
| Cursor started in a corner; minimized window divided by zero | centered cursor and zero-size guard |
| Malformed CLI values silently became zero | validate complete, finite, bounded numeric values |
| `float2` alignment and inverse-square-force comments were inaccurate | fix the comments; the force is a softened radial field |
| Hybrid graphics / driver issues blocked the entire final demo | explicit, labeled `--staged` fallback; interop remains the default |

## What was missing from the learning path

These are good **next exercises**, not prerequisites to enjoying the demo:

1. **Warps and coalescing.** In 04, explain why consecutive `threadIdx.x` values
   form a warp, why transpose writes are strided, and how `[32][33]` changes bank
   indices. Compare padded and unpadded shared tiles. Accesses, caches, banks,
   and occupancy all matter; on-chip storage isn't automatically faster code.
2. **Reduction constraints.** The grid-stride traversal in 03 is general; its
   shared-memory tree reduction is not. It requires a one-dimensional,
   power-of-two block of at most 256 threads. Try shuffle-based reduction, then
   compare with CUB before assuming the hand-written version is fastest.
3. **Floating-point correctness.** Addition is not associative. Compare absolute
   and relative error, intentionally inject NaNs, and make validation reject
   them. A checksum is useful for observation, not proof of correctness.
4. **Honest benchmarking.** Warm up, repeat measurements, report medians and
   device properties. Example 05 times kernel execution with events and end-to-
   end pipelines with synchronized wall-clock measurements. Its single pipeline
   comparison is illustrative, not a rigorous performance claim. Stream speedup
   depends on copy engines, transfer size, and kernel occupancy.
5. **Ownership and synchronization.** Never read a mapped VBO with GL, never
   cache its mapped pointer across unmaps, and don't register persistent
   read-modify-write positions as `WriteDiscard`. Launch error checks do not
   replace execution checks; synchronization/copy/unmap surfaces async errors.
6. **Profile before optimizing.** Try 128/256/512-thread blocks. Measure the
   CUDA kernel separately from GL fill rate, bloom, and map/unmap overhead.
   FPS in the title is a whole-frame measurement, **not CUDA event timing**.

## Validation

```bash
make check
python3 scripts/smoke_test.py --runner prime-run
# If CUDA works but NVIDIA OpenGL doesn't:
python3 scripts/smoke_test.py --staged

compute-sanitizer --tool memcheck --error-exitcode 1 ./build/test_physics
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/04_shared_memory
```

The physics test covers all scenes, awkward counts, attraction and repulsion,
finite state, screen bounds, speed caps, and a sentinel after the arrays. The
window smoke test checks CLI failures, all scenes, frame dimensions/counts, and
that captured output changes. It does not assert cross-GPU bit-identical images,
benchmark performance, or exercise interactive keys automatically.

## Visual design choices

**Orbit / Aurora / Ink** give the viewer three recognizable forms rather than
one undifferentiated rainbow cloud. A low-opacity sprite deposits colour into
HDR history, quarter-resolution separable blur adds a halo, and exposure/tone
mapping delays white clipping. A restrained navy background, cyan accent,
compact bitmap labels, and a hands-free tour make the project understandable
even before someone picks up the mouse.

The fields are art-directed analytic vector fields. There are no particle-particle
forces, incompressibility solver, collision grid, or physically accurate gravity.
Those would be worthwhile follow-up projects, but would obscure this course's
first lesson: GPU parallelism can be simple, immediate, and beautiful.
