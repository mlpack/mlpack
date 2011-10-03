/** @file cuda_two_point_kernel.cu
 *
 *  CUDA implementation of two point.
 *
 *  @author Bill March (march@gatech.edu)
 */

#include <cuda.h>
#include <stdio.h>

__device__ float PointDistanceSqr(float* query_point, float* reference_point) {

  float x_dist = query_point[0] - reference_point[0];
  float y_dist = query_point[1] - reference_point[1];
  float z_dist = query_point[2] - reference_point[2];
  
  return (x_dist * x_dist + y_dist * y_dist + z_dist * z_dist);

} // PointDistanceSqr()

__device__ void LoadReferencePoint(
  float *reference,
  int reference_point_id,
  float *reference_point_shared_mem) {

  int num_dimensions = 3;

  int i;
  int source_pos = reference_point_id * num_dimensions;
  int dest_pos = threadIdx.x * num_dimensions;
  for(i = 0; i < num_dimensions; i++, dest_pos++, source_pos++) {
    reference_point_shared_mem[dest_pos] = reference[source_pos];
  }

// why are we resetting these?
  source_pos = reference_point_id * num_dimensions;
  dest_pos = threadIdx.x * num_dimensions;
} // LoadReferencePoint

__device__ void TestPointPair(
  float *query_point,
  float *reference_point,
  int *local_sum, float lower_bound_sqr, float upper_bound_sqr) {

  float squared_distance = PointDistanceSqr(query_point, reference_point);
  
  *local_sum += (lower_bound_sqr <= squared_distance) 
                && (squared_distance <= upper_bound_sqr);

} // TestPointPair

__device__ void TileBaseCase(
  float *query_point,
  int num_reference_points_in_this_tile,
  float *reference_point_shared_mem,
  int *local_sum, float lower_bound_sqr, float upper_bound_sqr) {

  int num_dimensions = 3;

  int i;
  float *reference_point = reference_point_shared_mem;
  for(i = 0; i < num_reference_points_in_this_tile; i++,
      reference_point += num_dimensions) {
    
    TestPointPair(query_point, reference_point, local_sum,
                  lower_bound_sqr, upper_bound_sqr);
  
  } // for i
} // TileBaseCase


__global__ void TwoPointKernelOnDevice(
  float *query, int num_query_points,
  float *reference, int num_reference_points,
  int *two_point_sums_out,
  float lower_bound_sqr, float upper_bound_sqr) {

  int num_dimensions = 3;

  // The shared memory that is used to load a list of reference
  // points.
  float query_point_local_mem[3];
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
  int local_sum = 0;

  // The number of reference points in each round is the minimum of
  // the two quantities: the number of points that can be packed in
  // the shared memory and the number of threads available.
  int num_reference_points_per_tile =
    min(1024 / num_dimensions, blockDim.x);

  for(i = 0, tile = 0; i < num_reference_points;
      i += num_reference_points_per_tile, tile++) {

    // Each thread loads the specified number of points, and
    // synchronize all threads within this block before computing.
    int reference_point_id = tile * num_reference_points_per_tile + threadIdx.x;
    int num_reference_points_in_this_tile =
      min(num_reference_points - i, num_reference_points_per_tile);
    int ending_rpoint_id = tile * num_reference_points_per_tile +
                           num_reference_points_in_this_tile;

    if(reference_point_id < ending_rpoint_id) {
      LoadReferencePoint(reference, reference_point_id,
                         reference_point_shared_mem);
    }
    __syncthreads();

    // Accumulate the kernel sum contribution of the current tile.
    if(global_thread_id < num_query_points) {
      TileBaseCase(query_point_local_mem,
                   num_reference_points_in_this_tile,
                   reference_point_shared_mem, &local_sum, lower_bound_sqr, 
                   upper_bound_sqr);
    }

    // Synchronize all threads within this block before loading new
    // sets of points.
    __syncthreads();
  }

  if(global_thread_id < num_query_points) {
    two_point_sums_out[ global_thread_id ] = local_sum;
  }
}

extern "C" {

  void TwoPointKernelOnHost(
    double *query, int num_query_points,
    double *reference, int num_reference_points,
    int *two_point_sums_out, float lower_bound_sqr, float upper_bound_sqr) {

    int num_dimensions = 3;

    // Prepare to copy the points into single precision format on the
    // GPU.
    float *query_on_host = new float[
      num_dimensions * num_query_points ];
    float *query_on_device = NULL;
    float *reference_on_host = new float[
      num_dimensions * num_reference_points ];
    float *reference_on_device = NULL;
    int *two_point_sums_out_on_device = NULL;
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
          &two_point_sums_out_on_device, num_query_points * sizeof(float))) {
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
    TwoPointKernelOnDevice <<< num_blocks, num_threads_per_block >>>(
      query_on_device, num_query_points,
      reference_on_device, num_reference_points, two_point_sums_out_on_device,
      lower_bound_sqr, upper_bound_sqr);

    // Copy out the result from the device to the host.
    cudaMemcpy(
      two_point_sums_out, two_point_sums_out_on_device,
      num_query_points * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory.
    delete[] query_on_host;
    delete[] reference_on_host;
  }
}
