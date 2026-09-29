// -----------------------------------------------------------------------------
//  04 — Shared memory: the matrix transpose
// -----------------------------------------------------------------------------
//  Build & run:   make run-04_shared_memory
//
//  Global memory (VRAM) is far away. Shared memory is a small scratchpad
//  *inside each SM*, shared by every thread of a block, and roughly two orders
//  of magnitude faster. The catch: it is tiny (a few tens of KB per block) and
//  you must synchronise threads with __syncthreads().
//
//  A matrix transpose is the classic reason to reach for it. In the naive
//  version, reading a column means stride-n access: every thread in a warp
//  touches a different 128-byte cache line, so you get terrible coalescing.
//
//  The trick: load 32x32 tile contiguously (coalesced reads), park it in
//  shared memory, __syncthreads(), then write it out transposed (coalesced
//  writes). Both sides of the transfer are now perfectly aligned.
//
//  There is one more detail: the shared array is declared [32][33] — the extra
//  column offsets the rows so that the strided access pattern does not hammer
//  the same shared-memory bank. That "+1" is the whole trick and it is worth
//  internalising.
// -----------------------------------------------------------------------------

#include <cstdio>

#include "cuda_check.h"

constexpr int N = 1024;      // matrix is N x N
constexpr int TILE = 32;     // tile / block edge (32x32 = 1024 threads/block)

// --- naive: read strided, write coalesced (or vice versa, either way it hurts)
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
  const int x = blockIdx.x * TILE + threadIdx.x;  // column
  const int y = blockIdx.y * TILE + threadIdx.y;  // row
  if (x < n && y < n) {
    out[x * n + y] = in[y * n + x];   // consecutive threads read n floats apart
  }
}

// --- shared-memory tiled transpose
__global__ void transpose_tiled(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
  // +1 padding avoids shared-memory bank conflicts on the read-back.
  __shared__ float tile[TILE][TILE + 1];

  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;

  // Load a tile with coalesced global reads.
  if (x < n && y < n) {
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];
  }
  __syncthreads();   // nobody may read the tile before everyone has written it

  // Write it back transposed: swap the block coordinates as well as the
  // in-block coordinates, so consecutive threads write consecutive addresses.
  const int out_x = blockIdx.y * TILE + threadIdx.x;
  const int out_y = blockIdx.x * TILE + threadIdx.y;
  if (out_x < n && out_y < n) {
    out[out_y * n + out_x] = tile[threadIdx.x][threadIdx.y];
  }
}

int main() {
  const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

  auto* h_in = static_cast<float*>(std::malloc(bytes));
  auto* h_out = static_cast<float*>(std::malloc(bytes));
  for (int i = 0; i < N * N; ++i) h_in[i] = static_cast<float>(i);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  const dim3 block(TILE, TILE);
  const dim3 grid(N / TILE, N / TILE);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int reps = 20;
  auto time_it = [&](auto launch) {
    // warm-up, then time `reps` launches back to back
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / reps;
  };

  const float ms_naive = time_it([&] { transpose_naive<<<grid, block>>>(d_in, d_out, N); });
  const float ms_tiled = time_it([&] { transpose_tiled<<<grid, block>>>(d_in, d_out, N); });

  // --- correctness against a CPU transpose
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  int wrong = 0;
  for (int y = 0; y < N && wrong == 0; y += 97) {        // spot-check
    for (int x = 0; x < N; x += 89) {
      if (h_out[x * N + y] != h_in[y * N + x]) ++wrong;
    }
  }

  std::printf("matrix    : %d x %d floats (%.1f MB)\n", N, N, bytes / 1048576.0);
  std::printf("block     : %dx%d   grid: %dx%d\n", TILE, TILE, N / TILE, N / TILE);
  std::printf("naive     : %8.3f ms\n", ms_naive);
  std::printf("tiled     : %8.3f ms   (%.2fx faster)\n", ms_tiled,
              ms_naive / ms_tiled);
  std::printf("correct   : %s\n", wrong == 0 ? "yes" : "NO");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  std::free(h_in);
  std::free(h_out);
  return EXIT_SUCCESS;
}
