// -----------------------------------------------------------------------------
//  05 — Measuring time, and overlapping work with streams
// -----------------------------------------------------------------------------
//  Build & run:   make run-05_timing_streams
//
//  Two everyday tools:
//
//  * CUDA events — the right way to time GPU work. Wall-clock timers on the
//    host measure the *launch*, not the execution, because launches are async.
//
//  * Streams — independent queues of work. Put chunk A's copy, compute and
//    copy-back in stream 0, chunk B's in stream 1, and the GPU can do a
//    transfer while another stream is busy computing. This is the cheapest
//    speed-up available to almost any real program.
//
//  The catch: cudaMemcpyAsync only overlaps if the host memory is *pinned*
//  (page-locked). With ordinary pageable memory the driver quietly makes the
//  copy synchronous and you get exactly zero benefit.
// -----------------------------------------------------------------------------

#include <chrono>
#include <cmath>
#include <cstdio>

#include "cuda_check.h"

// A deliberately compute-heavy per-element kernel, so that moving data and
// crunching numbers are comparable in cost and overlap is measurable.
__global__ void heavy_kernel(const float* __restrict__ in,
                             const float* __restrict__ noise,
                             float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float v = in[i];
#pragma unroll 8
  for (int k = 0; k < 512; ++k) {
    v = fmaf(v, 1.0000001f, 1e-7f);   // dependent chain: real work, not a no-op
  }
  out[i] = v + noise[i];
}

static double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

int main() {
  const int CHUNKS = 4;
  const int CHUNK = 1 << 20;                 // 1,048,576 floats = 4 MB
  const int n = CHUNK * CHUNKS;
  const size_t chunk_bytes = static_cast<size_t>(CHUNK) * sizeof(float);

  // ---- host side: pinned (page-locked) memory ------------------------------
  float* h_in = nullptr;
  float* h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, chunk_bytes * CHUNKS));   // pinned
  CUDA_CHECK(cudaMallocHost(&h_out, chunk_bytes * CHUNKS));
  for (int i = 0; i < n; ++i) h_in[i] = 1.0f + 1e-6f * (i % 512);

  // ---- device side ---------------------------------------------------------
  float *d_in = nullptr, *d_noise = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, chunk_bytes * CHUNKS));
  CUDA_CHECK(cudaMalloc(&d_noise, chunk_bytes * CHUNKS));
  CUDA_CHECK(cudaMalloc(&d_out, chunk_bytes * CHUNKS));
  CUDA_CHECK(cudaMemset(d_noise, 0, chunk_bytes * CHUNKS));

  const int block = 256;
  const int grid = (CHUNK + block - 1) / block;

  // ===========================================================================
  //  A. Timing a single kernel with CUDA events
  // ===========================================================================
  std::printf("--- A. one kernel, timed with cudaEvent ---\n");
  {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int reps = 30;
    heavy_kernel<<<grid, block>>>(d_in, d_noise, d_out, CHUNK);  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r) {
      heavy_kernel<<<grid, block>>>(d_in, d_noise, d_out, CHUNK);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= reps;

    const double flops = 2.0 * CHUNK * 512;  // one FMA = 2 flops
    std::printf("  %.3f ms/kernel  ->  %.0f GFLOP/s\n", ms,
                flops / (ms * 1e-3) / 1e9);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  // ===========================================================================
  //  B. Serial: every chunk copies in, computes, copies out — all in order
  // ===========================================================================
  std::printf("\n--- B. %d chunks, serial on the default stream ---\n", CHUNKS);
  const double t_serial = now_ms();
  for (int c = 0; c < CHUNKS; ++c) {
    const size_t off = static_cast<size_t>(c) * CHUNK;
    CUDA_CHECK(cudaMemcpy(d_in + off, h_in + off, chunk_bytes,
                          cudaMemcpyHostToDevice));
    heavy_kernel<<<grid, block>>>(d_in + off, d_noise + off, d_out + off, CHUNK);
    CUDA_CHECK(cudaMemcpy(h_out + off, d_out + off, chunk_bytes,
                          cudaMemcpyDeviceToHost));
  }
  const double serial_ms = now_ms() - t_serial;
  std::printf("  %.2f ms\n", serial_ms);

  // ===========================================================================
  //  C. Pipelined: one stream per chunk, async copies + pinned memory
  // ===========================================================================
  std::printf("\n--- C. %d chunks, one stream each ---\n", CHUNKS);
  cudaStream_t streams[CHUNKS];
  for (int c = 0; c < CHUNKS; ++c) {
    CUDA_CHECK(cudaStreamCreate(&streams[c]));
  }

  const double t_pipe = now_ms();
  for (int c = 0; c < CHUNKS; ++c) {
    const size_t off = static_cast<size_t>(c) * CHUNK;
    cudaStream_t s = streams[c];
    CUDA_CHECK(cudaMemcpyAsync(d_in + off, h_in + off, chunk_bytes,
                               cudaMemcpyHostToDevice, s));
    heavy_kernel<<<grid, block, 0, s>>>(d_in + off, d_noise + off, d_out + off,
                                        CHUNK);
    CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, chunk_bytes,
                               cudaMemcpyDeviceToHost, s));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const double pipe_ms = now_ms() - t_pipe;
  std::printf("  %.2f ms   (%.2fx faster than serial)\n", pipe_ms,
              serial_ms / pipe_ms);

  // ---- sanity: both paths produce the same answer --------------------------
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += h_out[i];
  std::printf("\nchecksum: %.3f  (finite and stable means the overlap is correct)\n",
              sum);

  for (int c = 0; c < CHUNKS; ++c) CUDA_CHECK(cudaStreamDestroy(streams[c]));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_noise));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return EXIT_SUCCESS;
}
