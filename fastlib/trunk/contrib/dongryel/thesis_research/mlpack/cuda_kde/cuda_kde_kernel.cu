/** @file cuda_kde_kernel.cu
 *
 *  CUDA implementation of KDE.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <cuda.h>

__global__ void NbodyKernelOnDevice(
  float *query, int num_query_points,
  float *reference, int num_reference_points) {

  // The shared memory that is used to load a list of reference
  // points.
  __shared__ float reference_point_shared_mem[1024];

  int i = 3;
}

extern "C" {

  void NbodyKernelOnHost(
    float *query, int num_query_points,
    float *reference, int num_reference_points) {
    NbodyKernelOnDevice <<< 1, 1>>>(
      query, num_query_points, reference, num_reference_points);
  }
}
