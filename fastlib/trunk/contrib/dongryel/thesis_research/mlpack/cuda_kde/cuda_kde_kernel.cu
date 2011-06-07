/** @file cuda_kde_kernel.cu
 *
 *  CUDA implementation of KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <cuda.h>

__device__ void LoadReferencePoint_(
  int num_dimensions,
  float *reference,
  int reference_point_id) {

  extern __shared__ float reference_point_shared_mem[];

  int i;
  int source_pos = reference_point_id * num_dimensions;
  int dest_pos = threadIdx.x * num_dimensions;
  for(i = 0; i < num_dimensions; i++, dest_pos++, source_pos++) {
    reference_point_shared_mem[dest_pos] = reference[source_pos];
  }
}

__device__ void AccumulateReferencePointContribution_(
  int num_dimensions,
  float bandwidth,
  float *query_point,
  float *reference_point,
  float *local_sum) {

  int i;
  float squared_distance = 0.0;
  for(i = 0; i < num_dimensions; i++) {
    float diff = (query_point[i] - reference_point[i]) / bandwidth;
    squared_distance += diff * diff;
  }
  float kernel_value = expf(- squared_distance * 0.5);
  (*local_sum) += kernel_value;
}

__device__ void AccumulateTileContribution_(
  int num_dimensions,
  float bandwidth,
  float *query_point,
  float *local_sum) {

  extern __shared__ float reference_point_shared_mem[];

  int i;
  float *reference_point = reference_point_shared_mem;
  for(i = 0; i < blockDim.x; i++, reference_point += num_dimensions) {
    AccumulateReferencePointContribution_(
      num_dimensions, bandwidth, query_point, reference_point, local_sum);
  }
}

__global__ void NbodyKernelOnDevice(
  int num_dimensions,
  float bandwidth,
  float *query, int num_query_points,
  float *reference, int num_reference_points,
  float *kernel_sums_out) {

  // The shared memory that is used to load a list of reference
  // points.
  float query_point_local_mem[100];
  __shared__ float reference_point_shared_mem[1024];

  // Load the assigned query point to the local memory.
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = num_dimensions * global_thread_id;
  int i, tile;
  for(i = 0; i < num_dimensions; i++, offset++) {
    query_point_local_mem[i] = query[offset];
  }

  // Local variable for accumulating the kernel sum.
  float local_sum = 0.0;

  for(i = 0, tile = 0; i < num_reference_points; i += blockDim.x, tile++) {

    // Each thread loads the specified number of points, and
    // synchronize all threads within this block before computing.
    int reference_point_id = tile * blockDim.x + threadIdx.x;
    LoadReferencePoint_(num_dimensions, reference, reference_point_id);
    __syncthreads();

    // Accumulate the kernel sum contribution of the current tile.
    AccumulateTileContribution_(
      num_dimensions, bandwidth, query_point_local_mem, & local_sum);

    // Synchronize all threads within this block before loading new
    // sets of points.
    __syncthreads();
  }

  kernel_sums_out[ global_thread_id ] = local_sum;
}

extern "C" {

  void NbodyKernelOnHost(
    int num_dimensions,
    float bandwidth,
    float *query, int num_query_points,
    float *reference, int num_reference_points,
    float *kernel_sums_out) {

    // Query the number of multiprocessors on the GPU.
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(& count);
    cudaGetDeviceProperties(&prop, 0);

    int num_blocks = prop.multiProcessorCount;
    int num_threads_per_block = num_dimensions / num_blocks;
    NbodyKernelOnDevice <<< num_blocks, num_threads_per_block >>>(
      num_dimensions, bandwidth, query, num_query_points,
      reference, num_reference_points, kernel_sums_out);
  }
}
