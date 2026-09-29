// -----------------------------------------------------------------------------
//  03 — Index arithmetic and the grid-stride loop
// -----------------------------------------------------------------------------
//  Build & run:   make run-03_index_math
//
//  Two ideas that make CUDA suddenly tractable:
//
//  1. A 2-D problem (an image, a matrix) maps beautifully onto a 2-D grid.
//     Just remember the memory is still 1-D:  index = row * width + col.
//
//  2. If you do not know (or do not want to care about) how many threads the
//     hardware can run, write a "grid-stride loop": each thread handles
//     elements i, i+stride, i+2*stride, ... where stride = total threads.
//     The kernel then works for *any* size n, with any grid/block you like.
//     This is the single most portable pattern in CUDA.
// -----------------------------------------------------------------------------

#include <cstdio>

#include "cuda_check.h"

// -----------------------------------------------------------------------------
//  A. 2-D indexing: fill a matrix with a value that depends on (row, col).
// -----------------------------------------------------------------------------
__global__ void fill_matrix(float* __restrict__ m, int width, int height) {
  // A 2-D block gives us natural (x, y) coordinates.
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    // Two equivalent ways to say "which cell am I?" — same number:
    const int flat_1 = row * width + col;                          // row-major
    const int flat_2 = blockIdx.y * blockDim.y * width             //
                       + threadIdx.y * width                       //
                       + blockIdx.x * blockDim.x + threadIdx.x;    // expanded
    m[flat_1] = static_cast<float>(flat_1);
    // flat_2 == flat_1 by construction; kept to show the algebra.
    (void)flat_2;
  }
}

// -----------------------------------------------------------------------------
//  B. Grid-stride loop: sum an array of any length with a fixed grid.
//     No bounds check is even needed inside — the loop condition handles it.
// -----------------------------------------------------------------------------
__global__ void grid_stride_sum(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
  const int stride = gridDim.x * blockDim.x;
  int start = blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  for (int i = start; i < n; i += stride) {
    acc += in[i];
  }

  // Each thread now holds a partial sum. Reduce the block with shared memory
  // (a first taste of what example 04 is about).
  __shared__ float partial[256];
  partial[threadIdx.x] = acc;
  __syncthreads();

  // Tree reduction: 256 -> 128 -> 64 -> ... -> 1
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      partial[threadIdx.x] += partial[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Thread 0 of each block writes that block's contribution.
  if (threadIdx.x == 0) {
    out[blockIdx.x] = partial[0];
  }
}

int main() {
  const int width = 8;
  const int height = 6;
  const int cells = width * height;

  // ---------------------------------------------------------------------------
  //  Part A — 2-D indexing
  // ---------------------------------------------------------------------------
  std::printf("--- A. 2-D grid over a %dx%d matrix ---\n", width, height);

  float* d_m = nullptr;
  CUDA_CHECK(cudaMalloc(&d_m, cells * sizeof(float)));

  const dim3 block2d(4, 3);                                   // 4 x 3 = 12 threads
  const dim3 grid2d((width + 3) / 4, (height + 2) / 3);        // ceil per axis

  fill_matrix<<<grid2d, block2d>>>(d_m, width, height);
  CUDA_CHECK_KERNEL();

  float h_m[48];
  CUDA_CHECK(cudaMemcpy(h_m, d_m, cells * sizeof(float), cudaMemcpyDeviceToHost));
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      std::printf("%4.0f", h_m[row * width + col]);
    }
    std::printf("\n");
  }
  CUDA_CHECK(cudaFree(d_m));

  // ---------------------------------------------------------------------------
  //  Part B — grid-stride loop over a deliberately awkward size
  // ---------------------------------------------------------------------------
  const int n = 1'000'003;  // prime-ish: definitely not a multiple of the block
  std::printf("\n--- B. grid-stride sum of %d floats ---\n", n);

  float* h_in = static_cast<float*>(std::malloc(n * sizeof(float)));
  double reference = 0.0;
  for (int i = 0; i < n; ++i) {
    h_in[i] = 1.0f + 1e-6f * static_cast<float>(i % 1000);
    reference += h_in[i];
  }

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  const int block = 256;

  // A fixed, small grid: 64 blocks * 256 threads = 16,384 threads that will
  // chew through a million elements by striding. Bigger grids are not better;
  // "just enough blocks to fill the GPU" is the rule of thumb.
  const int grid = 64;
  CUDA_CHECK(cudaMalloc(&d_out, grid * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

  grid_stride_sum<<<grid, block>>>(d_in, d_out, n);
  CUDA_CHECK_KERNEL();

  float h_out[64];
  CUDA_CHECK(cudaMemcpy(h_out, d_out, grid * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // The per-block sums are still separate; finish the job on the CPU.
  double total = 0.0;
  for (int i = 0; i < grid; ++i) total += h_out[i];

  std::printf("  threads        : %d (grid %d x block %d)\n", grid * block, grid,
              block);
  std::printf("  elements/thread: ~%.1f\n", static_cast<double>(n) / (grid * block));
  std::printf("  gpu total      : %.4f\n", total);
  std::printf("  cpu reference  : %.4f\n", reference);
  std::printf("  difference     : %.4f  (float sums are not associative — tiny\n"
              "                            error is expected and fine)\n",
              total - reference);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  std::free(h_in);
  return EXIT_SUCCESS;
}
