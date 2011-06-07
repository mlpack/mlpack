/** @file cuda_kde_kernel.cu
 *
 *  CUDA implementation of KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <cuda.h>

__global__ void NbodyKernelOnDevice(
  int num_dimensions,
  float *query, int num_query_points,
  float *reference, int num_reference_points) {

  // The shared memory that is used to load a list of reference
  // points.
  float query_point_local_mem[10];
  __shared__ float reference_point_shared_mem[1024];

  // Load the assigned query point to the local memory.
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = num_dimensions * global_thread_id;
  int i;
  for(i = 0; i < num_dimensions; i++, offset++) {
    query_point_local_mem[i] = query[offset];
  }
}

extern "C" {

  void NbodyKernelOnHost(
    int num_dimensions,
    float *query, int num_query_points,
    float *reference, int num_reference_points) {

    // Query the number of multiprocessors on the GPU.
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(& count);
    cudaGetDeviceProperties(&prop, 0);

    int num_blocks = prop.multiProcessorCount;
    int num_threads_per_block = num_dimensions / num_blocks;
    NbodyKernelOnDevice <<< num_blocks, num_threads_per_block >>>(
      num_dimensions, query, num_query_points, reference, num_reference_points);
  }
}
