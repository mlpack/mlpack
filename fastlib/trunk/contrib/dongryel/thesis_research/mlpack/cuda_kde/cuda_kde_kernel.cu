/** @file cuda_kde_kernel.cu
 *
 *  CUDA implementation of KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <cuda.h>
#include <stdio.h>

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
  int num_reference_points_in_this_tile,
  float *local_sum) {

  extern __shared__ float reference_point_shared_mem[];

  int i;
  float *reference_point = reference_point_shared_mem;
  for(i = 0; i < num_reference_points_in_this_tile; i++,
      reference_point += num_dimensions) {
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
  if(global_thread_id < num_query_points) {
    for(i = 0; i < num_dimensions; i++, offset++) {
      query_point_local_mem[i] = query[offset];
    }
  }

  // Local variable for accumulating the kernel sum.
  float local_sum = 0.0;
  int num_reference_points_per_tile = 1024 / num_dimensions;

  for(i = 0, tile = 0; i < num_reference_points;
      i += num_reference_points_per_tile, tile++) {

    // Each thread loads the specified number of points, and
    // synchronize all threads within this block before computing.
    int reference_point_id = tile * num_reference_points_per_tile + threadIdx.x;
    int num_reference_points_in_this_tile =
      min(
        num_reference_points - i, num_reference_points_per_tile);
    int ending_rpoint_id = tile * num_reference_points_per_tile +
                           num_reference_points_in_this_tile;

    if(reference_point_id < ending_rpoint_id) {
      LoadReferencePoint_(num_dimensions, reference, reference_point_id);
    }
    __syncthreads();

    // Accumulate the kernel sum contribution of the current tile.
    if(global_thread_id < num_query_points) {
      AccumulateTileContribution_(
        num_dimensions, bandwidth, query_point_local_mem,
        num_reference_points_in_this_tile, & local_sum);
    }

    // Synchronize all threads within this block before loading new
    // sets of points.
    __syncthreads();
  }

  if(global_thread_id < num_query_points) {
    kernel_sums_out[ global_thread_id ] = local_sum;
  }
}

extern "C" {

  void NbodyKernelOnHost(
    int num_dimensions,
    float bandwidth,
    double *query, int num_query_points,
    double *reference, int num_reference_points,
    float *kernel_sums_out) {

    // Prepare to copy the points into single precision format on the
    // GPU.
    float *query_on_host = new float[
      num_dimensions * num_query_points ];
    float *query_on_device = NULL;
    float *reference_on_host = new float[
      num_dimensions * num_reference_points ];
    float *reference_on_device = NULL;
    float *kernel_sums_out_device = NULL;
    int num_query_bytes = num_query_points * num_dimensions * sizeof(float);
    int num_reference_bytes =
      num_reference_points * num_dimensions * sizeof(float);
    if(cudaSuccess != cudaMalloc(&query_on_device, num_query_bytes)) {
      printf("Error in allocating the query on the GPU.\n");
      return;
    }
    if(cudaSuccess != cudaMalloc(&reference_on_device, num_reference_bytes)) {
      printf("Error in allocating the reference on the GPU.\n");
      return;
    }
    if(cudaSuccess !=
        cudaMalloc(
          &kernel_sums_out_device, num_query_points * sizeof(float))) {
      printf("Error in allocating the kernel sum slots on the GPU.\n");
      return;
    }
    int i, j;
    int pos = 0;
    for(i = 0; i < num_query_points; i++) {
      for(j = 0; j < num_dimensions; j++, pos++) {
        query_on_host[pos] = query[pos];
      }
    }
    cudaMemcpy(
      query_on_device, query_on_host, num_query_bytes, cudaMemcpyHostToDevice);
    pos = 0;
    for(i = 0; i < num_reference_points; i++) {
      for(j = 0; j < num_dimensions; j++, pos++) {
        reference_on_host[pos] = reference[pos];
      }
    }
    cudaMemcpy(
      reference_on_device, reference_on_host,
      num_reference_bytes, cudaMemcpyHostToDevice);

    int num_threads_per_block = 512;
    int num_blocks = (num_query_points + num_threads_per_block - 1) /
                     num_threads_per_block;

    // Call the CUDA kernel.
    NbodyKernelOnDevice <<< num_blocks, num_threads_per_block >>>(
      num_dimensions, bandwidth, query_on_device, num_query_points,
      reference_on_device, num_reference_points, kernel_sums_out_device);

    // Copy out the result from the device to the host.
    cudaMemcpy(
      kernel_sums_out, kernel_sums_out_device,
      num_query_points * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory.
    delete[] query_on_host;
    delete[] reference_on_host;
  }
}
