// -----------------------------------------------------------------------------
//  01 — Hello, GPU
// -----------------------------------------------------------------------------
//  Build & run:   make run-01_hello
//
//  A CUDA program is ordinary C++ plus *kernels*: functions marked __global__
//  that run on the GPU. You launch one with angle-bracket syntax:
//
//        kernel<<<grid, block>>>(args...);
//
//  `grid` and `block` say how many threads to start and how they are grouped:
//
//        grid  = a 3-D array of blocks
//        block = a 3-D array of threads inside each block
//
//  So the total number of threads is grid.x*grid.y*grid.z * block.x*...*block.z.
//  Each thread finds out who it is through four built-in variables:
//
//        threadIdx  blockIdx  blockDim  gridDim      (each of them a uint3)
//
//  There is no "for" over your data: *you* are one of the threads. This one
//  idea — thread-per-element — is basically the whole programming model.
// -----------------------------------------------------------------------------

#include <cstdio>

#include "cuda_check.h"

// __global__ marks a kernel. It must return void, and it is called from the
// host but executed on the device, once per thread.
__global__ void hello_kernel() {
  // Every thread that reaches this line is a different "me".
  const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  printf("  block %2d  thread %2d  ->  global id %2d\n", blockIdx.x, threadIdx.x,
         global_id);
}

int main() {
  // ---------------------------------------------------------------------------
  //  1. Look at the device so you know what you are talking to.
  // ---------------------------------------------------------------------------
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "No CUDA-capable GPU found.\n");
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::printf("=====================================================\n");
  std::printf("  CUDA device 0\n");
  std::printf("=====================================================\n");
  std::printf("  name                 : %s\n", prop.name);
  std::printf("  compute capability   : %d.%d  (sm_%d%d)\n", prop.major,
              prop.minor, prop.major, prop.minor);
  std::printf("  multiprocessors (SMs): %d\n", prop.multiProcessorCount);
  std::printf("  threads per SM (max) : %d\n", prop.maxThreadsPerMultiProcessor);
  std::printf("  threads per block    : %d (max)\n", prop.maxThreadsPerBlock);
  std::printf("  shared mem per block : %.1f KB\n",
              prop.sharedMemPerBlock / 1024.0);
  std::printf("  global memory        : %.2f GB\n",
              prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  std::printf("  L2 cache             : %.1f MB\n", prop.l2CacheSize / 1048576.0);

  // ---------------------------------------------------------------------------
  //  2. Launch the kernel.
  //
  //     2 blocks x 4 threads = 8 threads in total, each printing one line.
  //     We keep it tiny on purpose: device printf is buffered and printed in
  //     whatever order the threads finish, so small grids stay readable.
  // ---------------------------------------------------------------------------
  const dim3 block(4);  // 4 threads per block
  const dim3 grid(2);   // 2 blocks

  std::printf("\nlaunching hello_kernel<<<%u, %u>>>() = %u threads\n\n",
              grid.x, block.x, grid.x * block.x);

  hello_kernel<<<grid, block>>>();

  // Kernel launches are *asynchronous*: the CPU does not wait. To see the
  // printf output we must synchronise, and this is also how we learn whether
  // the kernel actually succeeded.
  CUDA_CHECK_KERNEL();

  std::printf("\nThat is all a kernel is. Next: real data.\n");
  return EXIT_SUCCESS;
}
