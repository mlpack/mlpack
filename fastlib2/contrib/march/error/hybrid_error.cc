#include "hybrid_error.h"
#include "hybrid_error_stat.h"
#include "hybrid_error_analysis.h"
#include "naive_kernel_sum.h"
#include "fastlib/fastlib.h"
#include <stdlib.h>


int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  GaussianKernelErrorTester<AbsoluteErrorStat> absolute;
  GaussianKernelErrorTester<RelativeErrorStat> relative;
  GaussianKernelErrorTester<ExponentialErrorStat> exponential;
  GaussianKernelErrorTester<GaussianErrorStat> gaussian;
  GaussianKernelErrorTester<HybridErrorStat> hybrid;
  
  Matrix centers;
  
  const char* dataset = fx_param_str(NULL, "data", "test_data.csv");
  data::Load(dataset, &centers);
  
  double bandwidth = fx_param_double(NULL, "bandwidth", 0.1);
  DEBUG_ASSERT(bandwidth > 0.0);
  
  Vector abs_results;
  struct datanode* abs_mod = fx_submodule(NULL, "abs", "absolute");
  absolute.Init(abs_mod, centers, bandwidth);
  absolute.ComputeTotalSum(&abs_results);
  
  Vector rel_results;
  struct datanode* rel_mod = fx_submodule(NULL, "rel", "relative");
  relative.Init(rel_mod, centers, bandwidth);
  relative.ComputeTotalSum(&rel_results);
  
  Vector exp_results;
  struct datanode* exp_mod = fx_submodule(NULL, "exp", "exponential");
  exponential.Init(exp_mod, centers, bandwidth);
  exponential.ComputeTotalSum(&exp_results);
  
  Vector gauss_results;
  struct datanode* gauss_mod = fx_submodule(NULL, "gauss", "gaussian");
  gaussian.Init(gauss_mod, centers, bandwidth);
  gaussian.ComputeTotalSum(&gauss_results);
  
  Vector hybrid_results;
  struct datanode* hybrid_mod = fx_submodule(NULL, "hybrid", "hybrid");
  hybrid.Init(hybrid_mod, centers, bandwidth);
  hybrid.ComputeTotalSum(&hybrid_results);
  
  NaiveKernelSum naive;
  Vector naive_results;
  struct datanode* naive_mod = fx_submodule(NULL, "naive", "naive");
  naive.Init(naive_mod, centers, bandwidth);
  
  char output[50];
  strcpy(output, "naive_");
  char band_str[50];
  int success = sprintf(band_str, "%.2g", bandwidth);
  strcat(output, band_str);
  strcat(output, "_");
  strcat(output, dataset);
  
  Matrix naive_results_mat;
  
  if (data::Load(output, &naive_results_mat) == SUCCESS_FAIL) {
    printf("failed to load");
    naive.NaiveComputation(dataset, output, &naive_results);
  }
  else {
    printf("succeeded at load\n");
    naive_results_mat.MakeColumnVector(0, &naive_results);
  }
  
  // naive.ComputeTotalSum(&naive_results);
  
  
  ErrorAnalysis analysis;
  analysis.Init(abs_results, rel_results, exp_results, gauss_results, 
                hybrid_results, naive_results, abs_mod, rel_mod, exp_mod, 
                gauss_mod, hybrid_mod, naive_mod);
                
  analysis.ComputeResults();
  
  
 
  fx_done();

  return 0;

}