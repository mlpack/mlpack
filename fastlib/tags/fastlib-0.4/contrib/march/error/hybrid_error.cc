#include "hybrid_error.h"
#include "hybrid_error_stat.h"
#include "hybrid_error_analysis.h"
#include "naive_kernel_sum.h"
#include "fastlib/fastlib.h"
#include <stdlib.h>


int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);

  
  Matrix centers;
  
  const char* dataset = fx_param_str(NULL, "data", "test_data.csv");
  data::Load(dataset, &centers);
  
  double bandwidth = fx_param_double(NULL, "bandwidth", 0.1);
  DEBUG_ASSERT(bandwidth > 0.0);
  
  
  const char* kernel_name = fx_param_str_req(NULL, "kernel");
  
  double max_error = fx_param_double(NULL, "max_error", 0.1);
  
  double min_error = fx_param_double(NULL, "min_error", 0.01);
  
  double steepness = fx_param_double(NULL, "steepness", 0.1);
  
  Vector kernel_results;
  Vector naive_results;
  
  struct datanode* kernel_mod;
  struct datanode* naive_mod = fx_submodule(NULL, "naive");
  
  if (!strcmp(kernel_name, "abs")) {
    // max_error will just be epsilon, the others don't matter
    GaussianKernelErrorTester<AbsoluteErrorStat> absolute;
    kernel_mod = fx_submodule(NULL, "abs");
    absolute.Init(kernel_mod, centers, bandwidth, max_error, min_error, 
                  steepness);
    absolute.ComputeTotalSum(&kernel_results);
    
  } // abs
  else if (!strcmp(kernel_name, "rel")) {
    // max_error will just be epsilon, the others don't matter
    GaussianKernelErrorTester<RelativeErrorStat> relative;
    
    kernel_mod = fx_submodule(NULL, "rel");
    relative.Init(kernel_mod, centers, bandwidth, max_error, min_error, 
                  steepness);
    relative.ComputeTotalSum(&kernel_results);
    
  } // rel
  else if (!strcmp(kernel_name, "exp")) {
 
    GaussianKernelErrorTester<ExponentialErrorStat> exponential;
    
    kernel_mod = fx_submodule(NULL, "exp");
    exponential.Init(kernel_mod, centers, bandwidth, max_error, min_error, 
                     steepness);
    exponential.ComputeTotalSum(&kernel_results);
    
  } // exp
  else if (!strcmp(kernel_name, "gauss")) {
  
    GaussianKernelErrorTester<GaussianErrorStat> gaussian;
    
    kernel_mod = fx_submodule(NULL, "gauss");
    gaussian.Init(kernel_mod, centers, bandwidth, max_error, min_error, 
                  steepness);
    gaussian.ComputeTotalSum(&kernel_results);
    
  } // gauss
  else if (!strcmp(kernel_name, "hybrid")) {
    // max_error will be epsilon, others won't count
    
    GaussianKernelErrorTester<HybridErrorStat> hybrid;
    
    kernel_mod = fx_submodule(NULL, "hybrid");
    hybrid.Init(kernel_mod, centers, bandwidth, max_error, min_error, 
                steepness);
    hybrid.ComputeTotalSum(&kernel_results);
  
  } // hybrid
  else if (!strcmp(kernel_name, "naive")) {
  
    NaiveKernelSum naive_sum;
    naive_sum.Init(naive_mod, centers, bandwidth);
    naive_sum.NaiveComputation(dataset, "/dev/null", &naive_results);
    //ot::Print(naive_results);
    fx_done(NULL);
    return 0;

  }
  else {
    printf("Invalid choice for kernel\n");
    return 1;
  }
  
  
  char output[50];
  strcpy(output, "../../../../naive/dist1/");
//  strcpy(output, "naive/");
  char band_str[50];
//  strcat(output, dataset);
  char num_str[50];
  sprintf(num_str, "%d", centers.n_cols());
  strcat(output, num_str);
  strcat(output, "_");
  sprintf(band_str, "%.2g", bandwidth);
  strcat(output, band_str);
  
  Matrix naive_results_mat;
  
  if (data::Load(output, &naive_results_mat) == SUCCESS_FAIL) {
    // printf("failed to load");
    
    NaiveKernelSum naive;
    naive.Init(naive_mod, centers, bandwidth);
    naive.NaiveComputation(dataset, output, &naive_results);
  }
  else {
    // printf("succeeded at load\n");
    naive_results_mat.MakeColumnVector(0, &naive_results);
  }
  
  // naive.ComputeTotalSum(&naive_results);

  ErrorAnalysis analysis;
  analysis.Init(kernel_results, naive_results, kernel_mod, naive_mod);
                
  analysis.ComputeResults();
  
  //ot::Print(kernel_results);
  /*printf("naive_results\n");
  ot::Print(naive_results);
  */
 
  fx_done(NULL);

  return 0;

}
